"""Bounded, read-only dataset discovery through an existing pinned SSH client."""

from __future__ import annotations

import json
import math
from pathlib import PurePosixPath
import re

from .checkpoint_transfer_transport import TransferError, _remote_json


MAX_SCAN_RESULT_BYTES = 1024 * 1024
MAX_DATASETS = 200
MAX_DIRECTORIES = 5000
MAX_DEPTH = 8

# The browser supplies only JSON values, never command text. This script needs
# Python's standard library only, so scanning doesn't depend on a training env
# or on the version of the machine's running Agent.
_SCAN_SCRIPT = r'''
import errno, hashlib, json, math, os, pathlib, stat, sys, threading, time
MAX_INFO=1024*1024
MAX_TASKS=256*1024
MAX_TOTAL=8*1024*1024
MAX_RESULT=1024*1024
MAX_DIRS=5000
length=int(sys.stdin.buffer.readline(32))
if not 0<length<=65536:raise ValueError('Invalid dataset scan request')
p=json.loads(sys.stdin.buffer.read(length))
def disconnected():
    try:os.read(0,1)
    finally:os._exit(130)
threading.Thread(target=disconnected,daemon=True).start()
class ScanError(Exception):
    def __init__(self,code,message):self.code=code;self.message=message
def fail(code,message):raise ScanError(code,message)
def text(value,maximum=4096,multiline=False):
    if not isinstance(value,str) or not value or len(value)>maximum or any((ord(c)<32 and not(multiline and c in '\n\t')) or 127<=ord(c)<=159 for c in value):return None
    try:value.encode('utf-8')
    except UnicodeError:return None
    return value
def canonical(value):
    if not text(value) or '\\' in value or not value.startswith('/') or '..' in value.split('/'):
        fail('DATASET_SCAN_PATH_INVALID','Dataset parent must be an absolute path without traversal or control characters')
    return '/'+str(pathlib.PurePosixPath(value)).lstrip('/')
def under(path,root):
    path,root=pathlib.PurePosixPath(path),pathlib.PurePosixPath(root)
    return path==root or root in path.parents
def encode(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
def pairs(items):
    result={}
    for key,value in items:
        if key in result:raise ValueError('duplicate metadata key')
        result[key]=value
    return result
def reject(_):raise ValueError('non-finite metadata number')
def parse(value):return json.loads(value,object_pairs_hook=pairs,parse_constant=reject)
FLAGS=os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0)
def open_absolute(path):
    fd=os.open('/',FLAGS)
    try:
        for part in pathlib.PurePosixPath(path).parts[1:]:
            following=os.open(part,FLAGS,dir_fd=fd);os.close(fd);fd=following
        return fd
    except BaseException:os.close(fd);raise
def open_descendant(parent,parts):
    fd=os.dup(parent)
    try:
        for part in parts:
            following=os.open(part,FLAGS,dir_fd=fd);os.close(fd);fd=following
        return fd
    except BaseException:os.close(fd);raise
total_bytes=0
def read_file(fd,name,maximum):
    global total_bytes
    before=os.stat(name,dir_fd=fd,follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):fail('DATASET_SCAN_UNSAFE_METADATA','Dataset metadata must be an ordinary file, never a link or device')
    if before.st_size>maximum:fail('DATASET_SCAN_METADATA_LIMIT','Dataset metadata exceeds its read budget')
    if total_bytes+before.st_size>MAX_TOTAL:fail('DATASET_SCAN_TOTAL_LIMIT','Dataset scan reached its total metadata budget')
    opened=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|getattr(os,'O_NONBLOCK',0),dir_fd=fd)
    try:
        initial=os.fstat(opened)
        if (not stat.S_ISREG(initial.st_mode) or (before.st_dev,before.st_ino)!=(initial.st_dev,initial.st_ino)):
            fail('DATASET_SCAN_SOURCE_CHANGED','Dataset metadata changed while being opened')
        chunks=[];size=0
        while True:
            value=os.read(opened,min(65536,maximum+1-size))
            if not value:break
            size+=len(value);total_bytes+=len(value);chunks.append(value)
            if size>maximum or total_bytes>MAX_TOTAL:fail('DATASET_SCAN_METADATA_LIMIT','Dataset metadata exceeds its read budget')
        after=os.fstat(opened)
        if (initial.st_size,initial.st_mtime_ns,initial.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            fail('DATASET_SCAN_SOURCE_CHANGED','Dataset metadata changed while being read')
        return b''.join(chunks)
    finally:os.close(opened)
def metadata_summary(info,meta_fd):
    result={}
    fps=info.get('fps')
    if type(fps) in (int,float) and 0<fps<=1000 and math.isfinite(fps):result['fps']=fps
    for key in ('total_episodes','total_frames'):
        value=info.get(key)
        if type(value) is int and 0<=value<=2**63-1:result[key]=value
    robot=text(info.get('robot_type'),100) or text(info.get('robot_model'),100)
    if robot:result['robot_type']=robot
    task=text(info.get('task_description'),2000,True) or text(info.get('task'),2000,True)
    if not task:
        try:
            data=read_file(meta_fd,'tasks.jsonl',MAX_TASKS)
            for line in data.splitlines()[:8]:
                if len(line)>8192:break
                item=parse(line)
                if isinstance(item,dict):task=text(item.get('task'),2000,True)
                if task:break
        except ScanError as error:
            if error.code=='DATASET_SCAN_TOTAL_LIMIT':warn('task_budget','Some task summaries could not be read within the total metadata budget.',True)
        except (OSError,ValueError,TypeError,RecursionError,UnicodeError):pass
    if task:result['task_description']=task
    return result
def inspect(fd,path,relative):
    try:meta=os.stat('meta',dir_fd=fd,follow_symlinks=False)
    except FileNotFoundError:return None
    entry={'path':path,'relative_path':relative or '.', 'name':pathlib.PurePosixPath(path).name[:200] or 'dataset',
           'info_sha256':None,'feature_signature':None,'summary':{},'error':None}
    try:
        if not stat.S_ISDIR(meta.st_mode):fail('DATASET_SCAN_UNSAFE_METADATA','Dataset meta must be an ordinary directory, never a link')
        meta_fd=os.open('meta',FLAGS,dir_fd=fd)
        try:
            try:data=read_file(meta_fd,'info.json',MAX_INFO)
            except FileNotFoundError:return None
            entry['info_sha256']=hashlib.sha256(data).hexdigest()
            info=parse(data)
            features=info.get('features') if isinstance(info,dict) else None
            if not isinstance(features,dict) or not 1<=len(features)<=256:
                fail('DATASET_SCAN_INVALID_METADATA','LeRobot info.json requires a bounded nonempty features object')
            signature=[]
            for key,feature in sorted(features.items()):
                if not text(key,256) or not isinstance(feature,dict) or not text(feature.get('dtype'),64):
                    fail('DATASET_SCAN_INVALID_METADATA','LeRobot feature declarations are invalid')
                shape=feature.get('shape')
                if not isinstance(shape,list) or len(shape)>5 or any(type(n) is not int or not 0<n<=2**31-1 for n in shape):
                    fail('DATASET_SCAN_INVALID_METADATA','LeRobot feature shapes are invalid')
                signature.append({'key':key,'dtype':feature['dtype'],'shape':shape,'names':feature.get('names')})
            entry['feature_signature']=hashlib.sha256(encode(signature)).hexdigest()
            entry['summary']=metadata_summary(info,meta_fd)
        finally:os.close(meta_fd)
    except ScanError as error:entry['error']={'code':error.code,'message':error.message}
    except PermissionError:entry['error']={'code':'DATASET_SCAN_PERMISSION_DENIED','message':'Dataset metadata cannot be read by the SSH user'}
    except OSError:entry['error']={'code':'DATASET_SCAN_UNSAFE_METADATA','message':'Dataset metadata cannot be safely opened'}
    except (ValueError,TypeError,UnicodeError,RecursionError):entry['error']={'code':'DATASET_SCAN_INVALID_METADATA','message':'Dataset metadata is not supported bounded JSON'}
    return entry
try:
    if not hasattr(os,'O_NOFOLLOW') or not hasattr(os,'O_DIRECTORY'):fail('DATASET_SCAN_UNSUPPORTED','Safe directory descriptor support is required')
    path=canonical(p['path']);roots=[canonical(root) for root in p['roots']]
    if not any(under(path,root) for root in roots):fail('DATASET_SCAN_PATH_OUTSIDE_ROOTS','Dataset parent is outside allowed directories')
    recursive=p['recursive'];max_depth=p['max_depth'];max_datasets=p['max_datasets']
    if type(recursive) is not bool or type(max_depth) is not int or not 1<=max_depth<=8 or type(max_datasets) is not int or not 1<=max_datasets<=200:fail('DATASET_SCAN_INVALID_REQUEST','Dataset scan limits are invalid')
    if not recursive:max_depth=1
    deadline=time.monotonic()+45
    result={'path':path,'entries':[],'warnings':[],'truncated':False,'scanned_directories':0,'skipped_directories':0}
    warning_codes=set();result_bytes=4096;listed_entries=0
    def warn(code,message,truncated=False):
        if code not in warning_codes:result['warnings'].append(message);warning_codes.add(code)
        if truncated:result['truncated']=True
    root=open_absolute(path)
    try:
        pending=[((),0)]
        while pending:
            if listed_entries>=50000:warn('listing','Scan reached its 50000-directory-entry budget; results may be incomplete.',True);break
            if time.monotonic()>deadline:warn('deadline','Scan reached its time budget; choose a smaller parent folder to continue.',True);break
            if result['scanned_directories']>=MAX_DIRS:warn('directories','Scan reached the 5000-directory budget; results may be incomplete.',True);break
            parts,depth=pending.pop();display=str(pathlib.PurePosixPath(path).joinpath(*parts))
            try:fd=open_descendant(root,parts)
            except OSError:
                result['skipped_directories']+=1;warn('unreadable','Some unreadable, linked or changed directories were skipped.');continue
            try:
                result['scanned_directories']+=1
                entry=inspect(fd,display,'/'.join(parts))
                if entry is not None:
                    size=len(encode(entry))
                    if len(result['entries'])>=max_datasets:warn('datasets','Scan reached its dataset limit; choose a smaller parent folder to continue.',True);break
                    if result_bytes+size>MAX_RESULT-65536:warn('result','Scan reached its response-size budget; results may be incomplete.',True);break
                    result_bytes+=size;result['entries'].append(entry)
                    if entry['error'] and entry['error']['code']=='DATASET_SCAN_TOTAL_LIMIT':warn('metadata','Scan reached its 8 MiB metadata budget; results may be incomplete.',True);break
                    # Do not walk data, videos or checkpoints inside a dataset.
                    continue
                if depth>=max_depth and not recursive:continue
                children=[]
                with os.scandir(fd) as entries:
                    for child in entries:
                        listed_entries+=1
                        if listed_entries>=50000:warn('listing','Scan reached its 50000-directory-entry budget; results may be incomplete.',True);break
                        if time.monotonic()>deadline:warn('deadline','Scan reached its time budget; choose a smaller parent folder to continue.',True);break
                        if not text(child.name,255) or '\\' in child.name:result['skipped_directories']+=1;warn('unsafe_name','Names with control characters or invalid text were skipped.');continue
                        if child.name.startswith('.'):
                            if child.is_dir(follow_symlinks=False):result['skipped_directories']+=1;warn('hidden','Hidden directories were skipped.')
                            continue
                        try:
                            if child.is_symlink():
                                result['skipped_directories']+=1;warn('linked','Symbolic links were skipped.');continue
                            if child.is_dir(follow_symlinks=False):children.append(child.name)
                        except OSError:result['skipped_directories']+=1;warn('unreadable','Some unreadable, linked or changed directories were skipped.')
                        if len(children)+len(pending)>MAX_DIRS:
                            warn('queued','Scan reached its directory-listing budget; results may be incomplete.',True);break
                if children and depth>=max_depth:
                    if recursive:warn('depth','Scan reached its maximum depth; deeper directories were not scanned.',True)
                    continue
                pending.extend((parts+(name,),depth+1) for name in sorted(children,reverse=True))
            except OSError:
                result['skipped_directories']+=1;warn('unreadable','Some unreadable, linked or changed directories were skipped.')
            finally:os.close(fd)
    finally:os.close(root)
    if len(encode(result))>MAX_RESULT:fail('DATASET_SCAN_RESULT_LIMIT','Dataset scan response exceeded its budget')
    print(json.dumps({'result':result},ensure_ascii=False,separators=(',',':'),allow_nan=False))
except ScanError as error:print(json.dumps({'error':{'code':error.code,'message':error.message}}))
except PermissionError:print(json.dumps({'error':{'code':'DATASET_SCAN_PERMISSION_DENIED','message':'SSH user cannot read the selected parent directory'}}))
except FileNotFoundError:print(json.dumps({'error':{'code':'DATASET_SCAN_NOT_FOUND','message':'Selected dataset parent directory does not exist'}}))
except OSError:print(json.dumps({'error':{'code':'DATASET_SCAN_PATH_UNSAFE','message':'Selected path contains a link, special file, or unreadable directory'}}))
except (KeyError,ValueError,TypeError,UnicodeError,RecursionError):print(json.dumps({'error':{'code':'DATASET_SCAN_INVALID_REQUEST','message':'Dataset scan request or result is invalid'}}))
'''


def _path(value):
    if (not isinstance(value, str) or not 1 <= len(value) <= 4096 or not value.startswith('/')
            or '\\' in value or '..' in value.split('/')
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)):
        raise TransferError('DATASET_SCAN_PATH_INVALID', '请输入允许访问目录内的数据集父目录绝对路径')
    try:
        value.encode('utf-8')
    except UnicodeError:
        raise TransferError('DATASET_SCAN_PATH_INVALID', '数据集父目录包含无效文本') from None
    return '/' + str(PurePosixPath(value)).lstrip('/')


def _under(path, root):
    target, base = PurePosixPath(path), PurePosixPath(root)
    return target == base or base in target.parents


def _text(value, maximum, *, multiline=False):
    if (not isinstance(value, str) or not 1 <= len(value) <= maximum
            or any((ord(char) < 32 and not (multiline and char in '\n\t'))
                   or 127 <= ord(char) <= 159 for char in value)):
        return False
    try:
        value.encode('utf-8')
    except UnicodeError:
        return False
    return True


def _validated_result(result, parent, roots, max_datasets):
    fields = {'path', 'entries', 'warnings', 'truncated', 'scanned_directories', 'skipped_directories'}
    if (not isinstance(result, dict) or set(result) != fields or result['path'] != parent
            or type(result['truncated']) is not bool or type(result['scanned_directories']) is not int
            or not 0 <= result['scanned_directories'] <= MAX_DIRECTORIES
            or type(result['skipped_directories']) is not int or not 0 <= result['skipped_directories'] <= 100000
            or not isinstance(result['warnings'], list) or len(result['warnings']) > 20
            or any(not _text(item, 1000) for item in result['warnings'])
            or not isinstance(result['entries'], list) or len(result['entries']) > max_datasets):
        raise TransferError('DATASET_SCAN_INVALID_RESULT', '机器返回的数据集扫描结果无效')
    seen = set()
    for entry in result['entries']:
        if not isinstance(entry, dict) or set(entry) != {'path', 'relative_path', 'name', 'info_sha256', 'feature_signature', 'summary', 'error'}:
            raise TransferError('DATASET_SCAN_INVALID_RESULT', '机器返回的数据集条目无效')
        try:
            path = _path(entry['path'])
        except TransferError:
            raise TransferError('DATASET_SCAN_INVALID_RESULT', '数据集条目路径无效') from None
        if (path != entry['path'] or path in seen or not _under(path, parent)
                or not any(_under(path, root) for root in roots)
                or entry['relative_path'] != str(PurePosixPath(path).relative_to(parent))
                or not _text(entry['name'], 200)):
            raise TransferError('DATASET_SCAN_INVALID_RESULT', '数据集条目路径与扫描范围不一致')
        seen.add(path)
        for field in ('info_sha256', 'feature_signature'):
            if entry[field] is not None and (not isinstance(entry[field], str) or not re.fullmatch('[a-f0-9]{64}', entry[field])):
                raise TransferError('DATASET_SCAN_INVALID_RESULT', '数据集元信息指纹无效')
        summary = entry['summary']
        if not isinstance(summary, dict) or set(summary) - {'fps', 'robot_type', 'total_episodes', 'total_frames', 'task_description'}:
            raise TransferError('DATASET_SCAN_INVALID_RESULT', '数据集摘要包含不支持的内容')
        if ('robot_type' in summary and not _text(summary['robot_type'], 100)
                or 'task_description' in summary and not _text(summary['task_description'], 2000, multiline=True)
                or any(key in summary and (type(summary[key]) is not int or not 0 <= summary[key] <= 2**63 - 1)
                       for key in ('total_episodes', 'total_frames'))
                or 'fps' in summary and (type(summary['fps']) not in (int, float)
                                        or not 0 < summary['fps'] <= 1000 or not math.isfinite(summary['fps']))):
            raise TransferError('DATASET_SCAN_INVALID_RESULT', '数据集摘要值无效')
        if entry['error'] is not None and (not isinstance(entry['error'], dict) or set(entry['error']) != {'code', 'message'}):
            raise TransferError('DATASET_SCAN_INVALID_RESULT', '数据集错误摘要无效')
        if entry['error'] is not None and (not _text(entry['error']['message'], 500)
                or not isinstance(entry['error']['code'], str)
                or not re.fullmatch('DATASET_SCAN_[A-Z_]{1,80}', entry['error']['code'])):
            raise TransferError('DATASET_SCAN_INVALID_RESULT', '数据集错误摘要值无效')
        if entry['error'] is None and (entry['info_sha256'] is None or entry['feature_signature'] is None):
            raise TransferError('DATASET_SCAN_INVALID_RESULT', '数据集缺少元信息指纹')
    try:
        if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode()) > MAX_SCAN_RESULT_BYTES:
            raise ValueError
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise TransferError('DATASET_SCAN_INVALID_RESULT', '数据集扫描结果超过限制') from None
    return result


def scan_dataset_roots(remote, parent_path, roots, *, recursive=True, max_depth=1, max_datasets=MAX_DATASETS,
                       cancelled=None, timeout=50):
    """Discover metadata candidates without opening samples or altering files.

    ``remote`` must already be connected using the saved, pinned SSH identity.
    The caller owns the client and must close it after completion/cancellation.
    """
    parent = _path(parent_path)
    if not isinstance(roots, (list, tuple)) or not 1 <= len(roots) <= 100:
        raise TransferError('DATASET_SCAN_PATH_OUTSIDE_ROOTS', '机器允许访问目录未配置')
    roots = [_path(root) for root in roots]
    if not any(_under(parent, root) for root in roots):
        raise TransferError('DATASET_SCAN_PATH_OUTSIDE_ROOTS', '数据集父目录不在机器允许访问的目录内')
    if type(recursive) is not bool or type(max_depth) is not int or not 1 <= max_depth <= MAX_DEPTH or type(max_datasets) is not int or not 1 <= max_datasets <= MAX_DATASETS:
        raise TransferError('DATASET_SCAN_INVALID_REQUEST', '数据集扫描深度或数量限制无效')
    payload = {'path': parent, 'roots': roots, 'recursive': recursive,
               'max_depth': max_depth, 'max_datasets': max_datasets}
    if len(json.dumps(payload, ensure_ascii=False).encode()) > 65536:
        raise TransferError('DATASET_SCAN_INVALID_REQUEST', '数据集扫描范围超过请求限制')
    if type(timeout) not in (int, float) or not 0 < timeout <= 120 or not math.isfinite(timeout):
        raise TransferError('DATASET_SCAN_INVALID_REQUEST', '数据集扫描超时设置无效')
    try:
        result = _remote_json(remote, payload, cancelled or (lambda: False), timeout=timeout,
                              script=_SCAN_SCRIPT, max_bytes=MAX_SCAN_RESULT_BYTES)
        return _validated_result(result, parent, roots, max_datasets)
    except TransferError as error:
        codes = {'TRANSFER_CANCELLED': ('DATASET_SCAN_CANCELLED', '数据集扫描已取消'),
                 'TRANSFER_TIMEOUT': ('DATASET_SCAN_TIMEOUT', '数据集扫描超时，请选择更小的父目录'),
                 'TRANSFER_MANIFEST_LIMIT': ('DATASET_SCAN_RESULT_LIMIT', '数据集扫描结果超过限制'),
                 'TRANSFER_REMOTE_FAILED': ('DATASET_SCAN_REMOTE_FAILED', '机器无法完成数据集扫描，请检查 Python 和读取权限')}
        if error.code in codes:
            code, message = codes[error.code]
            raise TransferError(code, message) from None
        raise
