"""Machine-to-machine rsync pull with one narrowly scoped, short-lived SSH key.

The target keeps its generated private key.  Only the public key, pinned source
host key and bounded progress messages pass through the control plane.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import secrets
import shlex
import time

from .checkpoint_transfer_transport import TransferError, _open_channel


_MAX_MESSAGE = 32 * 1024

# Shared only by the fixed remote programs below.  No source text or shell
# arguments from a browser are evaluated on either machine.
_REMOTE_COMMON = r'''
import base64, contextlib, fcntl, hashlib, json, os, pathlib, pwd, re, select, shlex, shutil, signal, stat, subprocess, sys, tempfile, threading, time
def fail(code,message):raise ValueError(code+': '+message)
def send(value):
    print(json.dumps(value,separators=(',',':'),ensure_ascii=False,allow_nan=False),flush=True)
def request():
    deadline=time.monotonic()+60
    def read(count):
        result=b''
        while len(result)<count:
            remaining=deadline-time.monotonic()
            if remaining<=0 or not select.select([0],[],[],remaining)[0]:
                fail('DIRECT_TIMEOUT','Direct control setup exceeded its deadline')
            chunk=os.read(0,count-len(result))
            if not chunk:break
            result+=chunk
        return result
    line=b''
    while not line.endswith(b'\n') and len(line)<32:
        chunk=read(1)
        if not chunk:
            if not line:return None
            fail('DIRECT_PROTOCOL_INVALID','Incomplete control message')
        line+=chunk
    if not line.strip().isdigit():fail('DIRECT_PROTOCOL_INVALID','Invalid control message')
    count=int(line)
    if not 0<count<=32768:fail('DIRECT_PROTOCOL_INVALID','Control message exceeds its limit')
    raw=read(count)
    if len(raw)!=count:fail('DIRECT_PROTOCOL_INVALID','Incomplete control message')
    value=json.loads(raw)
    if not isinstance(value,dict):fail('DIRECT_PROTOCOL_INVALID','Invalid control message')
    return value
def path_fd(value,directory=True):
    if (not isinstance(value,str) or not value.startswith('/') or len(value)>4096
            or '..' in pathlib.PurePosixPath(value).parts or any(ord(c)<32 or 127<=ord(c)<=159 for c in value)):
        fail('DIRECT_PATH_INVALID','A canonical absolute path is required')
    normalized='/'+str(pathlib.PurePosixPath(value)).lstrip('/')
    if value!=normalized:fail('DIRECT_PATH_INVALID','Linked or noncanonical paths are not allowed')
    parts=pathlib.PurePosixPath(value).parts[1:]
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for index,part in enumerate(parts):
            flags=os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK
            if directory or index<len(parts)-1:flags|=os.O_DIRECTORY
            following=os.open(part,flags,dir_fd=fd)
            os.close(fd);fd=following
        info=os.fstat(fd)
        if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
            fail('DIRECT_PATH_INVALID','The asset path has an unsupported type')
        return fd
    except BaseException:
        os.close(fd);raise
def stamp(info):return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
def process_stamp(pid):
    try:
        text=pathlib.Path('/proc/'+str(pid)+'/stat').read_text()
        return text[text.rfind(')')+2:].split()[19]
    except (OSError,IndexError):return None
def alive(pid,identity):
    try:os.kill(pid,0)
    except OSError:return False
    return identity is None or process_stamp(pid)==identity
def remove_temp(path,identity):
    try:
        info=os.lstat(path)
        if (stat.S_ISDIR(info.st_mode) and info.st_uid==os.geteuid()
                and [info.st_dev,info.st_ino]==identity):shutil.rmtree(path)
    except FileNotFoundError:pass
def stop_child(child):
    if child is not None and child.poll() is None:
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait(timeout=3)
def error_result(error):
    code,separator,message=str(error).partition(': ')
    if not separator or not code.startswith('DIRECT_'):
        code,message='DIRECT_REMOTE_FAILED','Direct transfer preparation failed; check SSH access and filesystem permissions'
    return {'type':'error','error':{'code':code,'message':message[:1000]}}
'''


_AUTHORIZED_KEYS_HELPERS = r'''
def safe_ssh_directory(home):
    fd=path_fd(home)
    try:
        try:os.mkdir('.ssh',0o700,dir_fd=fd)
        except FileExistsError:pass
        ssh=os.open('.ssh',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    info=os.fstat(ssh)
    if info.st_uid!=os.geteuid() or info.st_mode&0o022:
        os.close(ssh);fail('DIRECT_AUTH_DIRECTORY_UNSAFE','Source SSH directory must be owned by the SSH user and not writable by others')
    return ssh
@contextlib.contextmanager
def auth_lock(ssh):
    # Keep one stable inode forever.  Separate exclusive creation from opening
    # an existing lock (concurrent O_CREAT itself can race on some filesystems).
    fd=None
    for attempt in range(5):
        try:
            fd=os.open('.termitech-direct.lock',os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_NONBLOCK,0o600,dir_fd=ssh)
        except FileExistsError:
            try:fd=os.open('.termitech-direct.lock',os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=ssh)
            except FileNotFoundError:time.sleep(.01);continue
        if fd is not None:break
    if fd is None:fail('DIRECT_AUTH_CHANGED','Source SSH authorization lock changed concurrently; retry the transfer')
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.geteuid() or info.st_nlink!=1 or info.st_mode&0o077:
            fail('DIRECT_AUTH_DIRECTORY_UNSAFE','Source SSH authorization lock is unsafe')
        fcntl.flock(fd,fcntl.LOCK_EX)
        yield
    finally:os.close(fd)
def read_authorized(ssh):
    try:fd=os.open('authorized_keys',os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=ssh)
    except FileNotFoundError:return b'',None
    with os.fdopen(fd,'rb') as stream:
        info=os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid!=os.geteuid() or info.st_nlink!=1
                or info.st_mode&0o022 or info.st_size>2*1024*1024):
            fail('DIRECT_AUTH_DIRECTORY_UNSAFE','Source authorized_keys must be a bounded, user-owned regular file')
        raw=stream.read(2*1024*1024+1)
        if len(raw)>2*1024*1024 or stamp(info)!=stamp(os.fstat(stream.fileno())):
            fail('DIRECT_AUTH_CHANGED','Source authorized_keys changed while reading')
    return raw,info
def replace_authorized(ssh,raw,previous):
    name='.termitech-authorized-'+os.urandom(12).hex()
    fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=ssh)
    try:
        with os.fdopen(fd,'wb') as stream:
            if previous is not None:os.fchmod(stream.fileno(),stat.S_IMODE(previous.st_mode))
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        try:current=os.stat('authorized_keys',dir_fd=ssh,follow_symlinks=False)
        except FileNotFoundError:current=None
        if ((previous is None)!=(current is None) or previous is not None and stamp(previous)!=stamp(current)):
            fail('DIRECT_AUTH_CHANGED','Source authorized_keys changed concurrently; retry the transfer')
        os.replace(name,'authorized_keys',src_dir_fd=ssh,dst_dir_fd=ssh);os.fsync(ssh)
    finally:
        try:os.unlink(name,dir_fd=ssh)
        except FileNotFoundError:pass
def install_authorized(ssh,line):
    with auth_lock(ssh):
        raw,info=read_authorized(ssh)
        separator=b'' if not raw or raw.endswith(b'\n') else b'\n'
        replace_authorized(ssh,raw+separator+line.encode()+b'\n',info)
        return {'length':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'separator':bool(separator)}
def remove_authorized(ssh,line,restore=None):
    with auth_lock(ssh):
        for attempt in range(3):
            raw,info=read_authorized(ssh)
            exact=line.encode()+b'\n'
            # Match one complete line, preserving all unrelated bytes and
            # concurrently added keys.  Never delete by comment or prefix.
            pieces=raw.splitlines(keepends=True)
            revised=b''.join(piece for piece in pieces if piece!=exact)
            if revised==raw:return
            if (restore and restore['separator'] and len(revised)==restore['length']+1
                    and revised.endswith(b'\n') and hashlib.sha256(revised[:-1]).hexdigest()==restore['sha256']):
                revised=revised[:-1]
            try:replace_authorized(ssh,revised,info);return
            except ValueError as error:
                if not str(error).startswith('DIRECT_AUTH_CHANGED:') or attempt==2:raise
'''


_FORCED_SENDER_SCRIPT = _REMOTE_COMMON + r'''
# There is deliberately no evaluation of SSH_ORIGINAL_COMMAND or rsync -s.
try:
    if len(sys.argv)!=1:fail('DIRECT_SOURCE_SCOPE_INVALID','Unexpected sender arguments')
    spec_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'scope.json')
    fd=os.open(spec_path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size>8192:raise ValueError('scope')
        scope=json.load(stream)
    source=scope['path'];kind=scope['kind']
    if time.time()>scope['deadline']:fail('DIRECT_SOURCE_SCOPE_EXPIRED','Temporary sender authorization expired')
    if kind=='directory':
        source_fd=path_fd(source)
        os.fchdir(source_fd)
        argument='./'
    else:
        parent,name=os.path.split(source)
        source_fd=path_fd(parent)
        info=os.stat(name,dir_fd=source_fd,follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):fail('DIRECT_SOURCE_SCOPE_INVALID','Source file changed type')
        os.fchdir(source_fd)
        argument='./'+name
    # Keep the directory FD across exec, so a renamed ancestor cannot redirect
    # the sender.  rsync -r does not dereference a replaced symlink file.
    os.set_inheritable(source_fd,True)
    binary=shutil.which('rsync')
    if not binary:fail('DIRECT_RSYNC_MISSING','rsync is not installed on the source machine')
    os.execv(binary,[binary,'--server','--sender','-rct','--partial','.',argument])
except BaseException:
    print('Scoped direct sender could not start',file=sys.stderr)
    sys.exit(126)
'''


_SOURCE_WATCHDOG_SCRIPT = _REMOTE_COMMON + _AUTHORIZED_KEYS_HELPERS + r'''
p=json.loads(sys.argv[1]);deadline=time.monotonic()+p['seconds']
while time.monotonic()<deadline and alive(p['parent'],p['parent_identity']):time.sleep(.5)
try:
    ssh=safe_ssh_directory(p['home'])
    try:remove_authorized(ssh,p['line'],p.get('restore'))
    finally:os.close(ssh)
finally:remove_temp(p['temp'],p['temp_identity'])
'''


_SOURCE_SCRIPT = _REMOTE_COMMON + _AUTHORIZED_KEYS_HELPERS + r'''
ssh=None;temporary=None;temporary_identity=None;line=None;watchdog=None;restore=None
try:
    p=request()
    if p is None:raise ValueError('missing request')
    home=pwd.getpwuid(os.geteuid()).pw_dir
    ssh=safe_ssh_directory(home)
    # Validate the path again at authorization time, and again in the forced
    # sender when the authenticated peer actually starts.
    source_fd=path_fd(p['path'],p['kind']=='directory');os.close(source_fd)
    temporary=tempfile.mkdtemp(prefix='.termitech-direct-'+p['token']+'-',dir=os.path.join(home,'.ssh'))
    info=os.lstat(temporary);temporary_identity=[info.st_dev,info.st_ino]
    for name,value in [('sender.py',SENDER_SCRIPT.encode()),('scope.json',json.dumps({'path':p['path'],'kind':p['kind'],'deadline':time.time()+p['auth_seconds']}).encode())]:
        fd=os.open(os.path.join(temporary,name),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as stream:stream.write(value)
    command=shlex.join([sys.executable,os.path.join(temporary,'sender.py')])
    escaped=command.replace('\\','\\\\').replace('"','\\"')
    # OpenSSH 8.x accepts local timestamps, but not the later Z suffix.
    expiry=time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()+p['auth_seconds']))
    line='restrict,expiry-time="'+expiry+'",command="'+escaped+'" '+p['public_key']+' termitech-direct-'+p['token']
    if len(line.encode())>8192:fail('DIRECT_AUTH_SCOPE_INVALID','Temporary authorized-key line exceeds the SSH limit')
    restore=install_authorized(ssh,line)
    spec={'home':home,'line':line,'temp':temporary,'temp_identity':temporary_identity,
          'parent':os.getpid(),'parent_identity':process_stamp(os.getpid()),'seconds':p['auth_seconds'],'restore':restore}
    watchdog=subprocess.Popen([sys.executable,'-c',WATCHDOG_SCRIPT,json.dumps(spec)],stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    time.sleep(.03)
    if watchdog.poll() is not None:fail('DIRECT_AUTH_WATCHDOG_FAILED','Source authorization expiry watcher did not start')
    send({'type':'ready'})
    closed=threading.Event()
    def eof():
        try:
            while os.read(0,1024):pass
        finally:closed.set()
    threading.Thread(target=eof,daemon=True).start()
    closed.wait(p['timeout']+30)
except BaseException as error:
    send(error_result(error))
finally:
    if ssh is not None:
        try:
            if line:remove_authorized(ssh,line,restore)
        except BaseException:
            send({'type':'cleanup_warning','message':'Temporary source authorization will expire automatically'})
        finally:os.close(ssh)
    if temporary:remove_temp(temporary,temporary_identity)
'''
_SOURCE_SCRIPT = ('SENDER_SCRIPT=' + repr(_FORCED_SENDER_SCRIPT) + '\nWATCHDOG_SCRIPT='
                  + repr(_SOURCE_WATCHDOG_SCRIPT) + '\n' + _SOURCE_SCRIPT)


_TARGET_WATCHDOG_SCRIPT = _REMOTE_COMMON + r'''
p=json.loads(sys.argv[1]);deadline=time.monotonic()+p['seconds']
while time.monotonic()<deadline and alive(p['parent'],p['parent_identity']):
    if not os.path.exists(p['temp']):sys.exit(0)
    time.sleep(.5)
try:
    # On Linux the recorded process start time prevents killing a reused PID.
    try:
        state=json.loads(pathlib.Path(p['temp'],'child.json').read_text())
        if (not alive(p['parent'],p['parent_identity']) and state.get('identity')
                and process_stamp(state['pid'])==state['identity']):
            os.killpg(state['pid'],signal.SIGTERM)
    except (OSError,ValueError,KeyError):pass
finally:remove_temp(p['temp'],p['temp_identity'])
'''


_TARGET_SCRIPT = _REMOTE_COMMON + r'''
def ssh_arguments(identity,keyfile,knownhosts,alias):
    return ['ssh','-F','/dev/null','-i',keyfile,'-p',str(identity['port']),'-l',identity['username'],
        '-o','HostKeyAlias='+alias,'-o','UserKnownHostsFile='+knownhosts,'-o','GlobalKnownHostsFile=/dev/null',
        '-o','StrictHostKeyChecking=yes','-o','CheckHostIP=no','-o','BatchMode=yes',
        '-o','IdentitiesOnly=yes','-o','IdentityAgent=none','-o','PasswordAuthentication=no',
        '-o','KbdInteractiveAuthentication=no','-o','PubkeyAuthentication=yes','-o','PreferredAuthentications=publickey',
        '-o','ControlMaster=no','-o','ControlPath=none','-o','ProxyCommand=none','-o','ProxyJump=none',
        '-o','PermitLocalCommand=no','-o','ClearAllForwardings=yes','-o','RequestTTY=no',
        '-o','ConnectTimeout=15','-o','ConnectionAttempts=1','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3']
temporary=None;temporary_identity=None;child=None;staging_fd=None;success=False
try:
    p=request()
    if p is None:raise ValueError('missing request')
    if any(shutil.which(binary) is None for binary in ('rsync','ssh','ssh-keygen')):
        fail('DIRECT_TOOLS_MISSING','Direct transfer requires rsync, ssh and ssh-keygen on the target machine')
    staging_fd=path_fd(p['staging'])
    temporary=tempfile.mkdtemp(prefix='.termitech-direct-'+p['token']+'-',dir='/tmp')
    info=os.lstat(temporary);temporary_identity=[info.st_dev,info.st_ino]
    watchdog_spec={'temp':temporary,'temp_identity':temporary_identity,'parent':os.getpid(),
        'parent_identity':process_stamp(os.getpid()),'seconds':min(600,p['timeout']+30)}
    watchdog=subprocess.Popen([sys.executable,'-c',WATCHDOG_SCRIPT,json.dumps(watchdog_spec)],stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    time.sleep(.03)
    if watchdog.poll() is not None:fail('DIRECT_KEY_WATCHDOG_FAILED','Target temporary-key watcher did not start')
    keyfile=os.path.join(temporary,'key')
    subprocess.run(['ssh-keygen','-q','-t','ed25519','-N','','-C','termitech-direct-'+p['token'],'-f',keyfile],
        stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True,timeout=20)
    os.chmod(keyfile,0o600)
    with open(keyfile+'.pub') as stream:public_key=' '.join(stream.read(8192).split()[:2])
    knownhosts=os.path.join(temporary,'known_hosts');alias='termitech-direct-'+p['token']
    fd=os.open(knownhosts,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as stream:stream.write(alias+' '+p['source']['key_type']+' '+p['source']['key_base64']+'\n')
    send({'type':'ready','public_key':public_key})
    ready=request()
    if ready!={'action':'start'}:fail('DIRECT_CANCELLED','Direct transfer cancelled before connection')
    disconnected=threading.Event()
    def eof():
        try:
            while os.read(0,1024):pass
        finally:disconnected.set()
    threading.Thread(target=eof,daemon=True).start()
    os.fchdir(staging_fd)
    rsh=shlex.join(ssh_arguments(p['source'],keyfile,knownhosts,alias))
    host=p['source']['host']
    origin=('['+host+']' if ':' in host else host)+':./'
    # Fixed source sender chooses its own path; client argv never carries a
    # dataset path through an intermediate login shell.
    child=subprocess.Popen(['rsync','-rct','--partial','--inplace','--progress','-e',rsh,'--',origin,'./'],
        stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,bufsize=0,
        env={**{key:value for key,value in os.environ.items() if not key.startswith('RSYNC_')},'LC_ALL':'C'})
    pathlib.Path(temporary,'child.json').write_text(json.dumps({'pid':child.pid,'identity':process_stamp(child.pid)}))
    os.set_blocking(child.stdout.fileno(),False)
    started=last_emit=time.monotonic();tail=b'';last=completed=0;done=p['initial_bytes'];rsync_output_bytes=0
    while True:
        chunk=b''
        if disconnected.is_set():fail('DIRECT_CANCELLED','Control connection closed; owned partial files are retained')
        elapsed=time.monotonic()-started
        if elapsed>p['timeout']:fail('DIRECT_TIMEOUT','Direct transfer exceeded its configured deadline')
        if select.select([child.stdout],[],[],.1)[0]:
            chunk=os.read(child.stdout.fileno(),65536)
            rsync_output_bytes+=len(chunk)
            if rsync_output_bytes>16*1024*1024:fail('DIRECT_OUTPUT_LIMIT','Direct rsync produced excessive diagnostic output')
            tail+=chunk
            pieces=re.split(rb'[\r\n]',tail);tail=pieces.pop()[-8192:]
            for text in pieces:
                match=re.match(rb'\s*([0-9][0-9,]*)\s+([0-9]{1,3})%',text)
                if match:
                    count=int(match.group(1).replace(b',',b''))
                    if count<last:completed+=last
                    last=count;done=min(p['bytes_total'],p['initial_bytes']+completed+count)
        if time.monotonic()-last_emit>=1:
            last_emit=time.monotonic()
            send({'type':'progress','event':{'stage':'TRANSFERRING','bytes_done':done,'bytes_total':p['bytes_total'],
                'files_done':None,'files_total':p['files_total'],'speed_bps':max(0,done-p['initial_bytes'])/max(elapsed,.001),
                'percent':min(99,100*done/p['bytes_total']) if p['bytes_total'] else 0}})
        if child.poll() is not None and not select.select([child.stdout],[],[],0)[0]:break
        if child.poll() is not None and not chunk:break
    if child.returncode:
        # Do not forward SSH diagnostics, paths, filenames or key material.
        fail('DIRECT_CONNECTION_UNAVAILABLE','Machines could not complete a direct SSH rsync connection; partial files are retained and no relay fallback was used')
    success=True
except BaseException as error:
    failure=error_result(error)
finally:
    try:stop_child(child)
    finally:
        if child is not None and child.stdout:child.stdout.close()
        if staging_fd is not None:os.close(staging_fd)
        if temporary:remove_temp(temporary,temporary_identity)
if success:send({'type':'complete','result':{'transport':'direct-ssh','bytes_total':p['bytes_total'],'files_total':p['files_total']}})
else:send(failure)
'''
_TARGET_SCRIPT = 'WATCHDOG_SCRIPT=' + repr(_TARGET_WATCHDOG_SCRIPT) + '\n' + _TARGET_SCRIPT


def _source_identity(remote):
    host, port, username = (getattr(remote, key, None) for key in ('host', 'port', 'username'))
    if (not isinstance(host, str) or len(host) > 253 or not re.fullmatch(r'[A-Za-z0-9_.:][A-Za-z0-9_.:-]*', host)
            or type(port) is not int or not 1 <= port <= 65535
            or not isinstance(username, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.-]{0,127}', username)):
        raise TransferError('DIRECT_SSH_IDENTITY_INVALID', 'Source SSH host, port or username is invalid')
    try:
        transport = remote.client.get_transport()
        if not transport.is_active():raise ValueError('inactive')
        key = transport.get_remote_server_key()
        key_type, encoded = key.get_name(), key.get_base64()
        raw = base64.b64decode(encoded, validate=True)
        if not raw or len(raw) > 16384 or not re.fullmatch(r'[A-Za-z0-9@._+-]{1,128}', key_type):raise ValueError('key')
        fingerprint = 'SHA256:' + base64.b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')
        pins = (getattr(remote, 'fingerprint', None), getattr(remote, 'expected_fingerprint', None))
        if not any(pins):
            raise TransferError('DIRECT_SSH_IDENTITY_INVALID', 'A previously verified source SSH host-key pin is required')
        for expected in pins:
            if expected and expected != fingerprint:
                raise TransferError('DIRECT_HOST_KEY_MISMATCH', 'The connected source host key differs from its pinned identity')
    except TransferError:
        raise
    except Exception:
        raise TransferError('DIRECT_SSH_IDENTITY_INVALID', 'An authenticated source SSH host key is required') from None
    return {'host': host, 'port': port, 'username': username, 'key_type': key_type, 'key_base64': encoded}


def _send(channel, value):
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
    if len(raw) > _MAX_MESSAGE:
        raise TransferError('DIRECT_PROTOCOL_INVALID', 'Direct transfer control request exceeds its limit')
    channel.sendall(str(len(raw)).encode() + b'\n' + raw)


class _Messages:
    def __init__(self, channel):
        self.channel, self.buffer, self.stderr = channel, bytearray(), 0
        self.total = 0

    def poll(self):
        while self.channel.recv_stderr_ready():
            chunk = self.channel.recv_stderr(4096)
            if not chunk:break
            self.stderr += len(chunk)
            if self.stderr > _MAX_MESSAGE:
                raise TransferError('DIRECT_OUTPUT_LIMIT', 'Direct transfer returned excessive diagnostics')
        while self.channel.recv_ready():
            chunk = self.channel.recv(4096)
            if not chunk:break
            self.buffer.extend(chunk)
            self.total += len(chunk)
            if len(self.buffer) > _MAX_MESSAGE or self.total > 64 * 1024 * 1024:
                raise TransferError('DIRECT_OUTPUT_LIMIT', 'Direct transfer progress exceeded its limit')
        result = []
        while b'\n' in self.buffer:
            raw, _, following = self.buffer.partition(b'\n')
            self.buffer[:] = following
            try:value = json.loads(raw)
            except (ValueError, UnicodeError):
                raise TransferError('DIRECT_PROTOCOL_INVALID', 'Direct transfer returned an invalid control message') from None
            if not isinstance(value, dict):
                raise TransferError('DIRECT_PROTOCOL_INVALID', 'Direct transfer returned an invalid control message')
            if value.get('type') == 'error':
                error = value.get('error', {})
                code = error.get('code')
                if not isinstance(code, str) or not re.fullmatch(r'DIRECT_[A-Z_]{1,80}', code):code = 'DIRECT_REMOTE_FAILED'
                # Fixed remote scripts provide messages; raw rsync stderr never
                # appears on this channel.
                message = error.get('message')
                if not isinstance(message, str):message = 'Direct transfer failed'
                raise TransferError(code, message[:1000])
            result.append(value)
        return result


def _finish(channel):
    output = bytearray()
    uncertain = False
    try:
        channel.shutdown_write()
        deadline = time.monotonic() + 5
        while not channel.exit_status_ready() and time.monotonic() < deadline:
            if channel.recv_ready():
                chunk = channel.recv(4096)
                if len(output) < _MAX_MESSAGE:output.extend(chunk[:_MAX_MESSAGE-len(output)])
            if channel.recv_stderr_ready():channel.recv_stderr(4096)
            time.sleep(.02)
        while channel.recv_ready() and len(output) < _MAX_MESSAGE:
            chunk = channel.recv(4096)
            if not chunk:break
            output.extend(chunk[:_MAX_MESSAGE-len(output)])
        uncertain = not channel.exit_status_ready()
    except Exception:
        uncertain = True
    finally:
        try:channel.close()
        except Exception:uncertain = True
    return uncertain or any(b'"cleanup_warning"' in line for line in output.splitlines())


def direct_copy(source_remote, target_remote, source_info, target_staging, transfer_id,
                progress=None, cancelled=None, *, initial_bytes=0, timeout=86400):
    """Pull directly from source to target; never silently fall back to a relay."""
    if not isinstance(transfer_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', transfer_id):
        raise TransferError('DIRECT_ID_INVALID', 'Invalid direct transfer identifier')
    if type(timeout) not in (int, float) or not 1 <= timeout <= 86400:
        raise TransferError('DIRECT_TIMEOUT_INVALID', 'Direct transfer deadline must be between 1 second and 24 hours')
    if (not isinstance(source_info, dict) or source_info.get('kind') not in {'directory', 'file'}
            or not isinstance(source_info.get('path'), str) or not isinstance(target_staging, str)):
        raise TransferError('DIRECT_PATH_INVALID', 'Direct transfer requires inspected source and staging paths')
    for path in (source_info['path'], target_staging):
        if (path == '/' or not path.startswith('/') or len(path) > 4096 or '..' in path.split('/')
                or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in path)):
            raise TransferError('DIRECT_PATH_INVALID', 'Invalid direct transfer path')
    manifest = source_info.get('manifest', {})
    totals = {key: manifest.get(key) for key in ('bytes_total', 'files_total')}
    if (any(type(value) is not int or value < 0 or value >= 2**63 for value in totals.values())
            or totals['files_total'] > 100000 or type(initial_bytes) is not int
            or not 0 <= initial_bytes <= totals['bytes_total']):
        raise TransferError('DIRECT_MANIFEST_INVALID', 'Invalid direct transfer totals')
    identity = _source_identity(source_remote)
    progress = progress or (lambda _event: None)
    cancelled = cancelled or (lambda: False)
    token, channels = secrets.token_hex(16), []
    setup_deadline = time.monotonic() + 60
    try:
        if cancelled():raise TransferError('TRANSFER_CANCELLED', 'Asset synchronization was cancelled')
        target = _open_channel(target_remote, shlex.join(['python3', '-u', '-c', _TARGET_SCRIPT]))
        channels.append(target)
        _send(target, {'token': token, 'transfer_id': transfer_id, 'source': identity,
                       'staging': target_staging, 'timeout': timeout, 'initial_bytes': initial_bytes, **totals})
        target_messages = _Messages(target)
        public_key = None
        while public_key is None:
            if cancelled():raise TransferError('TRANSFER_CANCELLED', 'Asset synchronization was cancelled')
            if time.monotonic() > setup_deadline:raise TransferError('DIRECT_TIMEOUT', 'Direct transfer setup timed out')
            for event in target_messages.poll():
                if event.get('type') != 'ready' or set(event) != {'type', 'public_key'}:
                    raise TransferError('DIRECT_PROTOCOL_INVALID', 'Unexpected target setup response')
                public_key = event['public_key']
                if not isinstance(public_key, str) or not re.fullmatch(r'ssh-ed25519 [A-Za-z0-9+/]{20,256}={0,2}', public_key):
                    raise TransferError('DIRECT_PROTOCOL_INVALID', 'Invalid temporary target public key')
            if target.exit_status_ready() and public_key is None:
                raise TransferError('DIRECT_CONNECTION_UNAVAILABLE', 'Target direct-transfer preparation stopped')
            time.sleep(.02)
        source = _open_channel(source_remote, shlex.join(['python3', '-u', '-c', _SOURCE_SCRIPT]))
        channels.append(source)
        _send(source, {'token': token, 'public_key': public_key, 'path': source_info['path'],
                       'kind': source_info['kind'], 'auth_seconds': 600, 'timeout': timeout})
        source_messages = _Messages(source)
        authorized = False
        while not authorized:
            if cancelled():raise TransferError('TRANSFER_CANCELLED', 'Asset synchronization was cancelled')
            if time.monotonic() > setup_deadline:raise TransferError('DIRECT_TIMEOUT', 'Direct transfer setup timed out')
            for event in source_messages.poll():
                if event != {'type': 'ready'}:
                    raise TransferError('DIRECT_PROTOCOL_INVALID', 'Unexpected source authorization response')
                authorized = True
            if source.exit_status_ready() and not authorized:
                raise TransferError('DIRECT_CONNECTION_UNAVAILABLE', 'Source direct-transfer authorization stopped')
            time.sleep(.02)
        progress({'stage': 'TRANSFERRING', 'message': '目标机器通过已固定主机密钥的 SSH 直接拉取源资产；文件内容不经过控制台',
                  'bytes_total': totals['bytes_total'], 'files_total': totals['files_total'], 'bytes_done': initial_bytes,
                  'files_done': None, 'speed_bps': 0, 'percent': 0})
        _send(target, {'action': 'start'})
        deadline = time.monotonic() + timeout + 5
        while True:
            if cancelled():raise TransferError('TRANSFER_CANCELLED', 'Asset synchronization was cancelled; partial files are retained')
            if time.monotonic() > deadline:raise TransferError('DIRECT_TIMEOUT', 'Direct transfer timed out; partial files are retained')
            for event in source_messages.poll():
                if event.get('type') != 'cleanup_warning':
                    raise TransferError('DIRECT_PROTOCOL_INVALID', 'Unexpected source control response')
            if source.exit_status_ready():
                raise TransferError('DIRECT_CONNECTION_UNAVAILABLE', 'Source control connection closed during direct transfer')
            for event in target_messages.poll():
                if event.get('type') == 'progress':
                    values = event.get('event')
                    if (not isinstance(values, dict) or values.get('stage') != 'TRANSFERRING'
                            or values.get('bytes_total') != totals['bytes_total'] or values.get('files_total') != totals['files_total']
                            or type(values.get('bytes_done')) is not int or not 0 <= values['bytes_done'] <= totals['bytes_total']
                            or values.get('files_done') is not None
                            or any(type(values.get(key)) not in (int, float) or not math.isfinite(values[key])
                                   or values[key] < 0 for key in ('speed_bps', 'percent'))
                            or values['percent'] > 99
                            or set(values) != {'stage', 'bytes_done', 'bytes_total', 'files_done', 'files_total', 'speed_bps', 'percent'}):
                        raise TransferError('DIRECT_PROTOCOL_INVALID', 'Invalid direct transfer progress')
                    progress(values)
                elif event.get('type') == 'complete':
                    expected = {'transport': 'direct-ssh', **totals}
                    if event.get('result') != expected:
                        raise TransferError('DIRECT_PROTOCOL_INVALID', 'Invalid direct transfer completion receipt')
                    return expected
                else:
                    raise TransferError('DIRECT_PROTOCOL_INVALID', 'Unexpected direct transfer response')
            if target.exit_status_ready():
                raise TransferError('DIRECT_CONNECTION_UNAVAILABLE', 'Target connection closed before direct transfer completed')
            time.sleep(.02)
    except (OSError, EOFError):
        raise TransferError('DIRECT_CONNECTION_UNAVAILABLE', 'Direct SSH control connection closed; partial files are retained') from None
    finally:
        # Stop the receiver before removing its source authorization.  EOF
        # invokes each supervisor's own finally block; detached watchers are
        # a second cleanup path if a process or the controller was killed.
        cleanup_warning = False
        for channel in channels:
            cleanup_warning = _finish(channel) or cleanup_warning
        if cleanup_warning:
            progress({'stage': 'TRANSFERRING', 'message': '临时 SSH 密钥或授权即时清理未确认；独立清理器和 10 分钟授权期限仍生效，请检查机器 SSH 授权状态'})
