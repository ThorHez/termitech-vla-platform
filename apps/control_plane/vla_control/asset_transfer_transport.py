"""Exact asset copies over direct machine SSH, with controller-side coordination."""
from __future__ import annotations

import re

from .checkpoint_transfer_transport import (
    TransferError, _remote_json, manifest_digest,
)


MAX_MANIFEST_FILES = 100_000
MAX_MANIFEST_BYTES = 64 * 1024 * 1024

_ASSET_REMOTE_SCRIPT = r'''
import ctypes, errno, hashlib, json, os, pathlib, shutil, socket, stat, sys, threading
MAX_FILES=100000
MAX_BYTES=64*1024*1024
length=int(sys.stdin.buffer.readline(32))
if not 0<length<=MAX_BYTES:raise ValueError('Transfer request exceeds its limit')
p=json.loads(sys.stdin.buffer.read(length))
def disconnected():
    try:os.read(0,1)
    finally:os._exit(130)
threading.Thread(target=disconnected,daemon=True).start()
def fail(code,message):raise ValueError(code+': '+message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def unsafe_name(name):return not name or any(ord(c)<32 or 127<=ord(c)<=159 for c in name)
def safe(value,roots,exists=True):
    path=pathlib.Path(value)
    if not path.is_absolute() or '..' in path.parts or unsafe_name(value):
        fail('TRANSFER_PATH_INVALID','Asset path must be an absolute path without traversal or control characters')
    resolved=path.resolve(strict=exists)
    if str(path)!=str(resolved):fail('TRANSFER_PATH_INVALID','Linked or noncanonical paths are not allowed')
    if not any(resolved==pathlib.Path(root).resolve() or pathlib.Path(root).resolve() in resolved.parents for root in roots):
        fail('TRANSFER_PATH_OUTSIDE_ROOTS','Asset path is outside machine managed directories')
    return resolved
def managed_mkdir(value,roots):
    path=safe(value,roots,False)
    bases=[pathlib.Path(root).resolve() for root in roots if path==pathlib.Path(root).resolve() or pathlib.Path(root).resolve() in path.parents]
    base=max(bases,key=lambda root:len(root.parts))
    fd=os.open(base,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0))
    try:
        for part in path.relative_to(base).parts:
            try:os.mkdir(part,mode=0o700,dir_fd=fd)
            except FileExistsError:pass
            following=os.open(part,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0),dir_fd=fd)
            os.close(fd);fd=following
    finally:os.close(fd)
    return safe(str(path),roots)
def component(name):
    if (not isinstance(name,str) or name in ('.','..') or len(name)>200 or unsafe_name(name)
            or '/' in name or '\\' in name or pathlib.Path(name).name!=name):
        fail('TRANSFER_PATH_INVALID','Destination name must be one safe directory name')
    return name
def machine_identity():
    try:value=pathlib.Path('/proc/sys/kernel/random/boot_id').read_bytes()
    except OSError:
        root=os.stat('/')
        value=(socket.gethostname()+':'+str(root.st_dev)+':'+str(root.st_ino)).encode()
    return hashlib.sha256(value).hexdigest()
def scan(path,source=False,exclude_train_state=False):
    before=path.lstat()
    if stat.S_ISREG(before.st_mode):
        kind='file';directories=[]
    elif stat.S_ISDIR(before.st_mode):
        kind='directory';directories=[]
    else:fail('TRANSFER_UNSAFE_FILE','Asset contains a linked or special file')
    files=[];manifest_bytes=512
    def read_file(name,display,fd=None):
        nonlocal manifest_bytes
        if unsafe_name(display):fail('TRANSFER_UNSAFE_FILE','Asset names cannot contain control characters')
        if len(files)>=MAX_FILES:fail('TRANSFER_MANIFEST_LIMIT','Asset exceeds the 100000-file limit')
        info=os.stat(name,dir_fd=fd,follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):fail('TRANSFER_UNSAFE_FILE','Asset contains a linked or special file')
        opened_fd=os.open(name,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0),dir_fd=fd)
        digest=hashlib.sha256()
        with os.fdopen(opened_fd,'rb') as reader:
            opened=os.fstat(reader.fileno())
            if not stat.S_ISREG(opened.st_mode) or (info.st_dev,info.st_ino)!=(opened.st_dev,opened.st_ino):
                fail('TRANSFER_SOURCE_CHANGED','Asset changed while opening a file')
            for chunk in iter(lambda:reader.read(4*1024*1024),b''):digest.update(chunk)
            after=os.fstat(reader.fileno())
        current=os.stat(name,dir_fd=fd,follow_symlinks=False)
        stamp=lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
        if stamp(info)!=stamp(after) or stamp(info)!=stamp(current):
            fail('TRANSFER_SOURCE_CHANGED','Asset changed while hashing')
        record={'path':display,'size':info.st_size,'sha256':digest.hexdigest()}
        manifest_bytes+=len(canonical(record))+1
        if manifest_bytes>MAX_BYTES-16384:fail('TRANSFER_MANIFEST_LIMIT','Asset manifest exceeds the 64 MiB limit')
        files.append(record)
    if kind=='file':read_file(str(path),path.name)
    else:
        for directory,dirs,children,fd in os.fwalk(path,follow_symlinks=False):
            relative=pathlib.Path(directory).relative_to(path)
            if str(relative)!='.':
                manifest_bytes+=len(canonical(str(relative)))+1
                if manifest_bytes>MAX_BYTES-16384:fail('TRANSFER_MANIFEST_LIMIT','Asset manifest exceeds the 64 MiB limit')
                directories.append(str(relative))
            if len(directories)>MAX_FILES:fail('TRANSFER_MANIFEST_LIMIT','Asset exceeds the 100000-directory limit')
            opened=os.fstat(fd);current=os.stat(directory,follow_symlinks=False)
            if (opened.st_dev,opened.st_ino)!=(current.st_dev,current.st_ino) or not stat.S_ISDIR(current.st_mode):
                fail('TRANSFER_SOURCE_CHANGED','Asset directory changed while scanning')
            for name in dirs:
                info=os.stat(name,dir_fd=fd,follow_symlinks=False)
                if unsafe_name(name) or not stat.S_ISDIR(info.st_mode):
                    fail('TRANSFER_UNSAFE_FILE','Asset contains a linked or special directory')
            # Only an ordinary directory at the asset root is excluded.  Check
            # its own type before pruning; a symlink never becomes authority
            # to bypass the normal safe-file contract.
            if source and exclude_train_state and str(relative)=='.' and 'train_state' in dirs:
                dirs.remove('train_state')
            for name in children:read_file(name,str(relative/name) if str(relative)!='.' else name,fd)
    end=path.lstat()
    if (before.st_dev,before.st_ino,before.st_mode)!=(end.st_dev,end.st_ino,end.st_mode):
        fail('TRANSFER_SOURCE_CHANGED','Asset root changed while scanning')
    files.sort(key=lambda item:item['path']);directories.sort()
    manifest={'schema_version':1,'files':files,'bytes_total':sum(item['size'] for item in files),'files_total':len(files)}
    result={'kind':kind,'name':path.name,'manifest':manifest,'directories':directories}
    if len(canonical(result))>MAX_BYTES-16384:fail('TRANSFER_MANIFEST_LIMIT','Asset manifest exceeds the 64 MiB limit')
    return result
def receipt_read(path):
    fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))
    with os.fdopen(fd,'rb') as reader:
        info=os.fstat(reader.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size>MAX_BYTES:
            fail('TRANSFER_PARTIAL_CONFLICT','Transfer receipt must be a bounded regular file')
        return json.load(reader)
def reject_excluded_target(path,exclude_train_state):
    if not exclude_train_state:return
    try:info=(path/'train_state').lstat()
    except FileNotFoundError:return
    if stat.S_ISDIR(info.st_mode):
        fail('TRANSFER_PARTIAL_CONFLICT','Excluded train_state directory is present in the owned destination; create a new transfer directory instead of reusing this partial copy')
    if not stat.S_ISREG(info.st_mode):
        fail('TRANSFER_UNSAFE_FILE','Destination contains a linked or special train_state entry')
def rename_noreplace(staging,final):
    libc=ctypes.CDLL(None,use_errno=True)
    left,right=os.fsencode(staging),os.fsencode(final)
    if sys.platform.startswith('linux') and hasattr(libc,'renameat2'):
        rename=libc.renameat2
        rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
        rename.restype=ctypes.c_int
        result=rename(-100,left,-100,right,1)
    elif sys.platform=='darwin' and hasattr(libc,'renamex_np'):
        rename=libc.renamex_np
        rename.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_uint]
        rename.restype=ctypes.c_int
        result=rename(left,right,4)
    else:fail('TRANSFER_ATOMIC_PUBLISH_UNSUPPORTED','Target filesystem lacks safe atomic publication')
    if result:
        code=ctypes.get_errno()
        if code in (errno.EEXIST,errno.ENOTEMPTY):fail('TRANSFER_TARGET_EXISTS','Destination already exists')
        if code in (errno.ENOSYS,errno.EINVAL,errno.EOPNOTSUPP):
            fail('TRANSFER_ATOMIC_PUBLISH_UNSUPPORTED','Target filesystem cannot atomically publish without replacing data')
        raise OSError(code,os.strerror(code))
try:
    mode=p['mode']
    exclude_train_state=p.get('exclude_train_state',False)
    if type(exclude_train_state) is not bool:
        fail('TRANSFER_PROFILE_INVALID','exclude_train_state must be a boolean')
    if mode=='source':
        root=safe(p['path'],p['roots'])
        if root==pathlib.Path('/'):fail('TRANSFER_PATH_INVALID','The filesystem root cannot be copied as an asset')
        result=scan(root,True,exclude_train_state);result['path']=str(root);result['machine_id']=machine_identity()
        if shutil.which('rsync') is None:fail('TRANSFER_RSYNC_MISSING','rsync is not installed on the source machine')
    elif mode=='prepare':
        parent=safe(p['parent'],p['roots'],False);name=component(p['name'])
        staging=parent/('.'+p['id']+'.partial');final=parent/name;receipt=parent/('.'+p['id']+'.asset-transfer.json')
        if final==staging or final==receipt:fail('TRANSFER_PATH_INVALID','Destination name conflicts with transfer metadata')
        if p['source_kind']=='directory' and machine_identity()==p['source_machine_id']:
            origin=pathlib.Path(p['source_identity']['path'])
            if any(value==origin or origin in value.parents or value in origin.parents for value in (staging,final)):
                fail('TRANSFER_PATH_OVERLAP','Source and destination overlap on the same machine')
        parent=managed_mkdir(p['parent'],p['roots'])
        ownership={'schema_version':1,'transfer_id':p['id'],'target_name':name,'source':p['source_identity'],
                   'source_kind':p['source_kind'],'source_name':p['source_name'],
                   'directories':p['directories'],'manifest':p['manifest']}
        if exclude_train_state:ownership['exclude_train_state']=True
        if len(canonical(ownership))>MAX_BYTES:fail('TRANSFER_MANIFEST_LIMIT','Transfer receipt exceeds the 64 MiB limit')
        if receipt.exists() or receipt.is_symlink():
            if receipt.is_symlink():fail('TRANSFER_PARTIAL_CONFLICT','Transfer receipt cannot be a symlink')
            previous=receipt_read(receipt)
            if not isinstance(previous,dict):fail('TRANSFER_SOURCE_CHANGED','Source asset or destination changed since this transfer began')
            if previous.get('exclude_train_state',False)!=exclude_train_state:
                fail('TRANSFER_PROFILE_CHANGED','The train_state exclusion profile changed; create a new transfer directory instead of reusing this copy')
            if previous!=ownership:fail('TRANSFER_SOURCE_CHANGED','Source asset or destination changed since this transfer began')
        else:
            if any(value.exists() or value.is_symlink() for value in (final,staging)):
                fail('TRANSFER_TARGET_EXISTS','Destination already exists and is not owned by this transfer')
            fd=os.open(receipt,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
            with os.fdopen(fd,'wb') as writer:
                writer.write(canonical(ownership));writer.flush();os.fsync(writer.fileno())
        expected_dirs=p['directories'] if p['source_kind']=='directory' else []
        if final.exists() or final.is_symlink():
            safe(str(final),p['roots']);reject_excluded_target(final,exclude_train_state);actual=scan(final)
            if actual['kind']=='directory' and actual['manifest']==p['manifest'] and actual['directories']==expected_dirs:
                result={'target_path':str(final),'published':True}
                print(json.dumps({'result':result},separators=(',',':'),ensure_ascii=False));sys.exit(0)
            fail('TRANSFER_TARGET_EXISTS','Published destination differs from this verified asset')
        if staging.is_symlink():fail('TRANSFER_PATH_INVALID','Partial destination cannot be a symlink')
        staging.mkdir(mode=0o700,exist_ok=True);safe(str(staging),p['roots'])
        reject_excluded_target(staging,exclude_train_state)
        existing=scan(staging)
        expected={item['path']:item for item in p['manifest']['files']}
        if (any(item['path'] not in expected for item in existing['manifest']['files'])
                or any(name not in expected_dirs for name in existing['directories'])):
            fail('TRANSFER_PARTIAL_CONFLICT','Partial destination contains unrelated files or directories')
        correct=sum(item['size'] for item in existing['manifest']['files'] if expected[item['path']]==item)
        required=max(0,p['manifest']['bytes_total']-correct)
        if shutil.disk_usage(parent).free<required+max(1024*1024,int(required*.02)):
            fail('TRANSFER_SPACE_INSUFFICIENT','Target has insufficient free storage for this asset')
        if shutil.which('rsync') is None:fail('TRANSFER_RSYNC_MISSING','rsync is not installed on the target machine')
        result={'staging_path':str(staging),'target_path':str(final),'bytes_existing':correct}
    elif mode=='inspect':
        inspected=safe(p['path'],p['roots']);reject_excluded_target(inspected,exclude_train_state);result=scan(inspected)
    elif mode=='publish':
        staging=safe(p['path'],p['roots']);final=safe(p['target'],p['roots'],False)
        if final.parent!=staging.parent or final.name!=component(p['name']) or staging.name!='.'+p['id']+'.partial':
            fail('TRANSFER_PATH_INVALID','Invalid asset publication destination')
        if final.exists() or final.is_symlink():fail('TRANSFER_TARGET_EXISTS','Destination already exists')
        reject_excluded_target(staging,exclude_train_state)
        # No reservation directory or check-then-replace window: interruption
        # leaves either the complete partial tree or the complete owned final.
        rename_noreplace(staging,final)
        fd=os.open(final.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
        try:os.fsync(fd)
        finally:os.close(fd)
        result={'target_path':str(final)}
    else:fail('TRANSFER_INTERNAL_ERROR','Unknown asset filesystem operation')
    print(json.dumps({'result':result},separators=(',',':'),ensure_ascii=False))
except Exception as error:
    code,separator,message=str(error).partition(': ')
    if not code.startswith('TRANSFER_') or not separator:
        code,message='TRANSFER_REMOTE_FAILED','Asset filesystem operation failed'
    print(json.dumps({'error':{'code':code,'message':message[:2000]}}))
'''


def _asset_json(remote, payload, cancelled):
    return _remote_json(remote, payload, cancelled, script=_ASSET_REMOTE_SCRIPT, max_bytes=MAX_MANIFEST_BYTES)


def transfer_asset(source_remote, target_remote, source_path, target_parent, transfer_id,
                   source_roots, target_roots, progress=None, cancelled=None, target_name=None,
                   exclude_train_state=False):
    """Copy a directory or single file into a new owned destination directory.

    Only bounded metadata and progress reach the controller; file bytes move
    directly from the source machine to the target machine.
    Retry reuses this task's partial directory; unrelated destinations are never adopted.
    """
    if type(exclude_train_state) is not bool:
        raise TransferError('TRANSFER_PROFILE_INVALID', 'exclude_train_state must be a boolean')
    if not isinstance(transfer_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', transfer_id):
        raise TransferError('TRANSFER_ID_INVALID', 'Invalid asset transfer identifier')
    name = transfer_id if target_name is None else target_name
    if (not isinstance(name, str) or not name or len(name) > 200 or name in ('.', '..')
            or '/' in name or '\\' in name or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in name)):
        raise TransferError('TRANSFER_PATH_INVALID', 'Destination name must be one safe directory name')
    progress = progress or (lambda _event: None)
    cancelled = cancelled or (lambda: False)
    profile = {'exclude_train_state': True} if exclude_train_state else {}
    progress({'stage': 'INSPECTING', 'message': '检查源资产文件与 SHA-256 清单'})
    source = _asset_json(source_remote, {'mode': 'source', 'path': source_path, 'roots': source_roots, **profile}, cancelled)
    original = source['manifest']
    progress({'stage': 'PREPARING', 'bytes_total': original['bytes_total'], 'files_total': original['files_total'],
              'message': '检查目标空间和本任务的可恢复目录'})
    prepared = _asset_json(target_remote, {'mode': 'prepare', 'parent': target_parent, 'roots': target_roots,
        'id': transfer_id, 'name': name, 'manifest': original, 'source_kind': source['kind'],
        'source_name': source['name'], 'directories': source['directories'], 'source_machine_id': source['machine_id'],
        'source_identity': {'path': source['path'], 'host': getattr(source_remote, 'host', None),
                            'port': getattr(source_remote, 'port', None),
                            'username': getattr(source_remote, 'username', None)}, **profile}, cancelled)
    if not prepared.get('published'):
        from .direct_asset_transfer import direct_copy
        direct_copy(source_remote, target_remote, source, prepared['staging_path'], transfer_id,
                    progress=progress, cancelled=cancelled, initial_bytes=prepared['bytes_existing'], **profile)
        progress({'stage': 'VERIFYING', 'bytes_total': original['bytes_total'], 'files_total': original['files_total'],
                  'message': '复核源数据稳定性和目标全部文件哈希'})
        after = _asset_json(source_remote, {'mode': 'source', 'path': source_path, 'roots': source_roots, **profile}, cancelled)
        if any(after[key] != source[key] for key in ('manifest', 'directories', 'kind', 'name', 'path')):
            raise TransferError('TRANSFER_SOURCE_CHANGED', 'Source asset changed during synchronization; partial files are retained')
        actual = _asset_json(target_remote, {'mode': 'inspect', 'path': prepared['staging_path'], 'roots': target_roots, **profile}, cancelled)
        if actual['manifest'] != original or actual['directories'] != source['directories']:
            raise TransferError('TRANSFER_CHECKSUM_MISMATCH', 'Target file hashes or directory layout differ from the source asset')
        _asset_json(target_remote, {'mode': 'publish', 'path': prepared['staging_path'], 'target': prepared['target_path'],
                                  'roots': target_roots, 'id': transfer_id, 'name': name, **profile}, cancelled)
    target = prepared['target_path']
    if cancelled():
        raise TransferError('TRANSFER_CANCELLED', 'Asset synchronization was cancelled; owned files are retained for retry')
    result = {'target_path': target, 'asset_path': target + '/' + source['name'] if source['kind'] == 'file' else target,
              'manifest_sha256': manifest_digest(original), 'manifest': original,
              'bytes_total': original['bytes_total'], 'files_total': original['files_total']}
    progress({'stage': 'SUCCEEDED', 'bytes_done': result['bytes_total'], 'bytes_total': result['bytes_total'],
              'files_done': result['files_total'], 'files_total': result['files_total'], 'speed_bps': 0, 'percent': 100})
    return result
