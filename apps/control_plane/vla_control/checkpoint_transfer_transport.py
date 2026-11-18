"""Checkpoint verification and publication around a direct machine-to-machine copy."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import shlex
import time


class TransferError(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(message)


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


_CHECKPOINT_EXCLUDED_ROOT_DIRECTORIES = ("train_state",)


_REMOTE_SCRIPT = r'''
import base64, hashlib, json, os, pathlib, shutil, stat, sys, threading
length=int(sys.stdin.buffer.readline(32))
if not 0<length<=16*1024*1024:
    raise ValueError('Transfer request exceeds its limit')
p = json.loads(sys.stdin.buffer.read(length))
# Keep stdin open while hashing. Channel cancellation gives this thread EOF,
# ending a long filesystem scan without leaving an orphan verification job.
def disconnected():
    try:os.read(0,1)
    finally:os._exit(130)
threading.Thread(target=disconnected,daemon=True).start()
def fail(code, message):
    raise ValueError(code + ': ' + message)
def safe(value, roots, exists=True):
    path = pathlib.Path(value)
    if not path.is_absolute() or '..' in path.parts or any(ord(c)<32 or ord(c)==127 for c in value):
        fail('TRANSFER_PATH_INVALID', 'Invalid absolute path')
    resolved = path.resolve(strict=exists)
    if str(path) != str(resolved):
        fail('TRANSFER_PATH_INVALID', 'Symlink or noncanonical path is not allowed')
    if not any(resolved == pathlib.Path(root).resolve() or pathlib.Path(root).resolve() in resolved.parents for root in roots):
        fail('TRANSFER_PATH_OUTSIDE_ROOTS', 'Checkpoint path is outside managed directories')
    return resolved
def managed_mkdir(value, roots):
    path=safe(value,roots,exists=False)
    bases=[pathlib.Path(root).resolve() for root in roots if path==pathlib.Path(root).resolve() or pathlib.Path(root).resolve() in path.parents]
    base=max(bases,key=lambda root:len(root.parts))
    descriptor=os.open(base,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0))
    try:
        for part in path.relative_to(base).parts:
            try:os.mkdir(part,mode=0o700,dir_fd=descriptor)
            except FileExistsError:pass
            following=os.open(part,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0),dir_fd=descriptor)
            os.close(descriptor);descriptor=following
    finally:os.close(descriptor)
    return safe(str(path),roots)
def manifest(path):
    files = []
    for directory, dirs, names in os.walk(path, followlinks=False):
        if pathlib.Path(directory) == path:
            dirs[:] = [name for name in dirs if not (name in __CHECKPOINT_EXCLUDED_ROOT_DIRECTORIES__
                and stat.S_ISDIR((path/name).lstat().st_mode))]
        for name in dirs:
            value = pathlib.Path(directory)/name
            if (any(ord(char)<32 or ord(char)==127 for char in str(value.relative_to(path)))
                    or value.is_symlink() or not stat.S_ISDIR(value.lstat().st_mode)):
                fail('TRANSFER_UNSAFE_FILE', 'Checkpoint contains a linked or special directory')
        for name in names:
            value = pathlib.Path(directory)/name
            relative=str(value.relative_to(path))
            if any(ord(char)<32 or ord(char)==127 for char in relative):
                fail('TRANSFER_UNSAFE_FILE','Checkpoint file names cannot contain control characters')
            info = value.lstat()
            if not stat.S_ISREG(info.st_mode):
                fail('TRANSFER_UNSAFE_FILE', 'Checkpoint contains a linked or special file')
            if len(files) >= 20000:
                fail('TRANSFER_MANIFEST_LIMIT', 'Checkpoint contains too many files')
            digest = hashlib.sha256()
            descriptor = os.open(value, os.O_RDONLY | getattr(os,'O_NOFOLLOW',0) | getattr(os,'O_NONBLOCK',0))
            with os.fdopen(descriptor,'rb') as reader:
                opened = os.fstat(reader.fileno())
                if not stat.S_ISREG(opened.st_mode) or (opened.st_dev,opened.st_ino)!=(info.st_dev,info.st_ino):
                    fail('TRANSFER_SOURCE_CHANGED', 'Checkpoint changed while opening a file')
                for chunk in iter(lambda:reader.read(4*1024*1024), b''):
                    digest.update(chunk)
                after = os.fstat(reader.fileno())
            if (info.st_size,info.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                fail('TRANSFER_SOURCE_CHANGED', 'Checkpoint changed while hashing')
            files.append({'path':relative, 'size':info.st_size, 'sha256':digest.hexdigest()})
    files.sort(key=lambda item:item['path'])
    return {'schema_version':1,'files':files,'bytes_total':sum(item['size'] for item in files),'files_total':len(files)}
def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def inference_manifest(value):
    # An earlier owned partial may include optimizer state. Ignore only that
    # old subtree; every retained file, source identity and sidecar must match.
    if not isinstance(value,dict) or set(value)!={'schema_version','files','bytes_total','files_total'} or value['schema_version']!=1:
        return None
    files=value['files']
    if (not isinstance(files,list) or any(not isinstance(item,dict) or set(item)!={'path','size','sha256'}
            or not isinstance(item['path'],str) or type(item['size']) is not int or item['size']<0 for item in files)
            or value['files_total']!=len(files) or value['bytes_total']!=sum(item['size'] for item in files)):
        return None
    kept=[item for item in files if not any(item['path'].startswith(name+'/') for name in __CHECKPOINT_EXCLUDED_ROOT_DIRECTORIES__)]
    return dict(value,files=kept,bytes_total=sum(item['size'] for item in kept),files_total=len(kept))
try:
    mode = p['mode']
    if mode == 'source':
        root = safe(p['path'],p['roots'])
        if not root.is_dir() or not ((root/'params').is_dir() or (root/'model.safetensors').is_file()) or not (root/'assets').is_dir():
            fail('TRANSFER_CHECKPOINT_INVALID','Checkpoint requires model weights and assets directory')
        result = {'kind':'directory','path':str(root),'manifest':manifest(root)}
        if not any(item['path'].endswith('norm_stats.json') and item['path'].startswith('assets/') for item in result['manifest']['files']):
            fail('TRANSFER_NORM_MISSING','Checkpoint assets contain no normalization statistics')
        if shutil.which('rsync') is None:
            fail('TRANSFER_RSYNC_MISSING','rsync is not installed on the source machine')
    elif mode == 'prepare':
        parent = managed_mkdir(p['parent'],p['roots'])
        if not parent.is_dir():
            fail('TRANSFER_TARGET_INVALID','Target parent must be an existing managed directory')
        staging = parent/('.'+p['id']+'.partial')
        final = parent/p['id']
        receipt = parent/('.'+p['id']+'.transfer.json')
        ownership = {'schema_version':1,'transfer_id':p['id'],'source':p['source_identity'],
                     'source_manifest':p['source_manifest'],'manifest':p['manifest']}
        if receipt.exists() or receipt.is_symlink():
            if receipt.is_symlink():
                fail('TRANSFER_PARTIAL_CONFLICT','Transfer receipt cannot be a symlink')
            descriptor = os.open(receipt,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))
            with os.fdopen(descriptor,'rb') as reader:
                info=os.fstat(reader.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size>16*1024*1024:
                    fail('TRANSFER_PARTIAL_CONFLICT','Transfer receipt is not a bounded regular file')
                previous=json.load(reader)
            if isinstance(previous,dict):
                previous=dict(previous)
                for key in ('source_manifest','manifest'):
                    previous[key]=inference_manifest(previous.get(key))
            if previous != ownership:
                fail('TRANSFER_SOURCE_CHANGED','Source checkpoint or inference configuration changed since this transfer began')
        else:
            # No receipt means there is no adoption authority for a directory
            # already present under this id, even if its content looks similar.
            if final.exists() or final.is_symlink() or staging.exists() or staging.is_symlink():
                fail('TRANSFER_PARTIAL_CONFLICT','Existing destination has no matching transfer receipt')
            descriptor=os.open(receipt,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
            with os.fdopen(descriptor,'wb') as writer:
                writer.write(canonical(ownership));writer.flush();os.fsync(writer.fileno())
        if final.exists() or final.is_symlink():
            safe(str(final),p['roots'])
            if final.is_dir() and manifest(final)==p['manifest']:
                result={'target_path':str(final),'published':True}
                print(json.dumps({'result':result},separators=(',',':'),ensure_ascii=False))
                sys.exit(0)
            fail('TRANSFER_TARGET_EXISTS','Published destination exists but is not the verified transferred checkpoint')
        if staging.is_symlink():
            fail('TRANSFER_PATH_INVALID','Partial destination cannot be a symlink')
        staging.mkdir(mode=0o700,exist_ok=True)
        safe(str(staging),p['roots'])
        existing = manifest(staging)
        expected = {item['path']:item for item in p['manifest']['files']}
        if any(item['path'] not in expected for item in existing['files']):
            fail('TRANSFER_PARTIAL_CONFLICT','Partial directory contains unrelated files')
        matching = [item for item in existing['files'] if expected[item['path']]==item]
        correct = sum(item['size'] for item in matching)
        source_paths = {item['path'] for item in p['source_manifest']['files']}
        source_bytes_existing = sum(item['size'] for item in matching if item['path'] in source_paths)
        # Reserve space for any file whose contents are not already verified.
        required = max(0,p['manifest']['bytes_total']-correct)
        margin = max(1024*1024,int(required*.02))
        if shutil.disk_usage(parent).free < required+margin:
            fail('TRANSFER_SPACE_INSUFFICIENT','Target has insufficient free storage for the checkpoint')
        if shutil.which('rsync') is None:
            fail('TRANSFER_RSYNC_MISSING','rsync is not installed on the target machine')
        result = {'staging_path':str(staging),'target_path':str(final),'source_bytes_existing':source_bytes_existing}
    elif mode == 'extras':
        staging = safe(p['path'],p['roots'])
        for name, encoded in p['files'].items():
            destination = staging/name
            contents=base64.b64decode(encoded)
            if destination.exists() or destination.is_symlink():
                descriptor=os.open(destination,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))
                with os.fdopen(descriptor,'rb') as reader:
                    info=os.fstat(reader.fileno())
                    if not stat.S_ISREG(info.st_mode) or info.st_size>1024*1024 or reader.read(1024*1024+1)!=contents:
                        fail('TRANSFER_EXTRA_CONFLICT','Portable inference sidecar already exists with different content')
                continue
            descriptor = os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
            with os.fdopen(descriptor,'wb') as writer:
                writer.write(contents);writer.flush();os.fsync(writer.fileno())
        result = {'written':True}
    elif mode == 'inspect':
        result = {'manifest':manifest(safe(p['path'],p['roots']))}
    elif mode == 'publish':
        staging = safe(p['path'],p['roots'])
        final = safe(p['target'],p['roots'],exists=False)
        if final.parent != staging.parent or final.name != p['id'] or staging.name != '.'+p['id']+'.partial':
            fail('TRANSFER_PATH_INVALID','Invalid publish destination')
        if final.exists() or final.is_symlink():
            fail('TRANSFER_TARGET_EXISTS','Published destination already exists')
        # mkdir creates the unique empty destination without ever replacing an
        # existing user directory; rename on POSIX replaces only this empty dir.
        final.mkdir(mode=0o700)
        try:
            os.rename(staging,final)
            descriptor=os.open(final.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
        except BaseException:
            final.rmdir()
            raise
        result = {'target_path':str(final)}
    else:
        fail('TRANSFER_INTERNAL_ERROR','Unknown transfer operation')
    print(json.dumps({'result':result},separators=(',',':'),ensure_ascii=False))
except Exception as error:
    text = str(error)
    code, separator, message = text.partition(': ')
    if not code.startswith('TRANSFER_') or not separator:
        code, message = 'TRANSFER_REMOTE_FAILED', 'Checkpoint filesystem operation failed'
    print(json.dumps({'error':{'code':code,'message':message[:2000]}}))
'''.replace('__CHECKPOINT_EXCLUDED_ROOT_DIRECTORIES__', repr(_CHECKPOINT_EXCLUDED_ROOT_DIRECTORIES))


def _open_channel(remote, command):
    if remote.client is None:
        raise TransferError("TRANSFER_SSH_UNAVAILABLE", "SSH connection is not available")
    transport = remote.client.get_transport()
    if transport is None or not transport.is_active():
        raise TransferError("TRANSFER_SSH_UNAVAILABLE", "SSH connection is not active")
    channel = transport.open_session(timeout=15)
    channel.settimeout(1)
    channel.exec_command(command)
    return channel


def _remote_json(remote, payload, cancelled, timeout=3600, *, script=None, max_bytes=16*1024*1024):
    channel = _open_channel(remote, shlex.join(["python3", "-u", "-c", _REMOTE_SCRIPT if script is None else script]))
    output, errors = bytearray(), bytearray()
    deadline = time.monotonic() + timeout
    try:
        if cancelled():
            raise TransferError("TRANSFER_CANCELLED", "Checkpoint transfer was cancelled")
        encoded=json.dumps(payload, ensure_ascii=False).encode()
        if len(encoded)>max_bytes:
            raise TransferError("TRANSFER_MANIFEST_LIMIT", "Checkpoint verification request exceeded its limit")
        channel.sendall(str(len(encoded)).encode()+b"\n"+encoded)
        while True:
            if cancelled():
                raise TransferError("TRANSFER_CANCELLED", "Checkpoint transfer was cancelled")
            if time.monotonic() > deadline:
                raise TransferError("TRANSFER_TIMEOUT", "Checkpoint filesystem verification timed out")
            for ready, receive, buffer in ((channel.recv_ready,channel.recv,output),
                                           (channel.recv_stderr_ready,channel.recv_stderr,errors)):
                while ready():
                    chunk=receive(65536)
                    if not chunk:break
                    buffer.extend(chunk)
                    if len(buffer)>max_bytes:
                        raise TransferError("TRANSFER_MANIFEST_LIMIT", "Remote verification result exceeded its limit")
            # SSH may deliver the process exit status before the remaining
            # stdout/stderr. A temporarily empty receive buffer is not EOF.
            if ((channel.eof_received or channel.closed) and channel.exit_status_ready()
                    and not channel.recv_ready() and not channel.recv_stderr_ready()):
                status=channel.recv_exit_status()
                if status:
                    raise TransferError("TRANSFER_REMOTE_FAILED", "Remote checkpoint verification process failed")
                try:value=json.loads(output)
                except (ValueError,UnicodeError):
                    raise TransferError("TRANSFER_REMOTE_FAILED", "Remote checkpoint verification returned an invalid result") from None
                if not isinstance(value,dict):
                    raise TransferError("TRANSFER_REMOTE_FAILED", "Remote checkpoint verification returned an invalid result")
                if value.get("error"):
                    error=value["error"]
                    raise TransferError(error.get("code","TRANSFER_REMOTE_FAILED"),error.get("message","Remote verification failed"))
                return value["result"]
            time.sleep(.01)
    finally:
        channel.close()


def transfer_checkpoint(source_remote, target_remote, source_path, target_parent, transfer_id,
                        source_roots, target_roots, progress=None, cancelled=None, extra_files=None):
    """Pull a checkpoint directly to a unique target directory, then verify and publish.

    ``remote`` objects must be connected SSHRemote instances. Callback receives
    plain dictionaries; no credential or raw remote stderr is forwarded to it.
    A failed retry reuses ``.<transfer_id>.partial`` and never deletes files.
    """
    if not isinstance(transfer_id,str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}",transfer_id):
        raise TransferError("TRANSFER_ID_INVALID", "Invalid checkpoint transfer identifier")
    progress=progress or (lambda _event:None)
    cancelled=cancelled or (lambda:False)
    extras={}
    for name,value in (extra_files or {}).items():
        if not isinstance(name,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",name):
            raise TransferError("TRANSFER_EXTRA_INVALID", "Inference sidecar must use a safe top-level filename")
        if isinstance(value,str):value=value.encode()
        if not isinstance(value,bytes):
            raise TransferError("TRANSFER_EXTRA_INVALID", "Inference sidecar must contain text or bytes")
        extras[name]=value
    if sum(map(len,extras.values()))>1024*1024:
        raise TransferError("TRANSFER_EXTRA_INVALID", "Inference sidecars exceed the 1 MiB limit")
    progress({"stage":"INSPECTING","message":"检查推理 checkpoint 的源文件哈希（已排除根目录 train_state/）"})
    source=_remote_json(source_remote,{"mode":"source","path":source_path,"roots":source_roots},cancelled)
    original=source["manifest"]
    if any(item["path"] in extras for item in original["files"]):
        raise TransferError("TRANSFER_EXTRA_CONFLICT", "Source checkpoint already contains the inference sidecar filename")
    expected=json.loads(json.dumps(original))
    for name,value in extras.items():
        expected["files"].append({"path":name,"size":len(value),"sha256":hashlib.sha256(value).hexdigest()})
    expected["files"].sort(key=lambda item:item["path"])
    expected.update(bytes_total=sum(item["size"] for item in expected["files"]),files_total=len(expected["files"]))
    progress({"stage":"PREPARING","bytes_total":expected["bytes_total"],"files_total":expected["files_total"],
              "message":"Checking target storage and recoverable partial directory"})
    prepared=_remote_json(target_remote,{"mode":"prepare","parent":target_parent,"roots":target_roots,
        "id":transfer_id,"manifest":expected,"source_manifest":original,
        "source_identity":{"path":source["path"],"host":getattr(source_remote,"host",None),
                           "port":getattr(source_remote,"port",None),"username":getattr(source_remote,"username",None)}},cancelled)
    if prepared.get("published"):
        result={"target_path":prepared["target_path"],"manifest_sha256":manifest_digest(expected),"manifest":expected,
                "bytes_total":expected["bytes_total"],"files_total":expected["files_total"]}
        progress({"stage":"SUCCEEDED","bytes_done":result["bytes_total"],"bytes_total":result["bytes_total"],
            "files_done":result["files_total"],"files_total":result["files_total"],"speed_bps":0,"percent":100})
        return result

    def report_copy_progress(event):
        if "bytes_total" in event:
            # The deployment includes sidecars that rsync does not copy.
            event = dict(event, bytes_total=expected["bytes_total"], files_total=expected["files_total"],
                         percent=min(99, 100 * event["bytes_done"] / max(1, expected["bytes_total"])))
        progress(event)

    from .direct_asset_transfer import direct_copy
    direct_copy(source_remote, target_remote, source, prepared["staging_path"], transfer_id,
                progress=report_copy_progress, cancelled=cancelled, initial_bytes=prepared["source_bytes_existing"],
                exclude_train_state=True)
    progress({"stage":"VERIFYING","bytes_done":original["bytes_total"],
              "bytes_total":expected["bytes_total"],"files_total":expected["files_total"],
              "message":"Comparing source stability and target SHA-256 manifest"})
    after=_remote_json(source_remote,{"mode":"source","path":source_path,"roots":source_roots},cancelled)["manifest"]
    if after!=original:
        raise TransferError("TRANSFER_SOURCE_CHANGED", "Source checkpoint changed during transfer; choose a completed checkpoint")
    if extras:
        # rsync leaves previous successful extra files unchanged during retry.
        current=_remote_json(target_remote,{"mode":"inspect","path":prepared["staging_path"],"roots":target_roots},cancelled)["manifest"]
        existing={item["path"]:item for item in current["files"]}
        expected_extra={item["path"]:item for item in expected["files"] if item["path"] in extras}
        if any(name in existing and existing[name]!=expected_extra[name] for name in extras):
            raise TransferError("TRANSFER_EXTRA_CONFLICT", "Existing partial sidecar differs from this transfer's configuration")
        pending={name:base64.b64encode(value).decode() for name,value in extras.items() if name not in existing}
        if pending:_remote_json(target_remote,{"mode":"extras","path":prepared["staging_path"],"roots":target_roots,"files":pending},cancelled)
    actual=_remote_json(target_remote,{"mode":"inspect","path":prepared["staging_path"],"roots":target_roots},cancelled)["manifest"]
    if actual!=expected:
        raise TransferError("TRANSFER_CHECKSUM_MISMATCH", "Target file hashes differ from the inference checkpoint")
    _remote_json(target_remote,{"mode":"publish","path":prepared["staging_path"],"target":prepared["target_path"],
        "roots":target_roots,"id":transfer_id},cancelled)
    result={"target_path":prepared["target_path"],"manifest_sha256":manifest_digest(expected),"manifest":expected,
            "bytes_total":expected["bytes_total"],"files_total":expected["files_total"]}
    progress({"stage":"SUCCEEDED","bytes_done":result["bytes_total"],"bytes_total":result["bytes_total"],
        "files_done":result["files_total"],"files_total":result["files_total"],"speed_bps":0,"percent":100})
    return result
