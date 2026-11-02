"""Read two bounded metadata JSON files through an existing pinned SSH client."""

import json
import math
from pathlib import PurePosixPath

from .checkpoint_transfer_transport import TransferError, _remote_json
from .dataset_scan_transport import _path, _under


MAX_INFO_BYTES = 1024 * 1024
MAX_NORM_BYTES = 8 * 1024 * 1024
MAX_RESULT_BYTES = 10 * 1024 * 1024

# Paths are delivered as framed JSON stdin, never interpolated into a shell.
# Descriptor-relative O_NOFOLLOW opens exclude links and races outside roots.
# This uses only Python's standard library; no Agent/training code is imported.
_READ_SCRIPT = r'''
import errno,hashlib,json,os,pathlib,stat,sys,threading
def disconnected():
    try:os.read(0,1)
    finally:os._exit(130)
def reject(_):raise ValueError('nonfinite JSON')
def pairs(items):
    result={}
    for key,value in items:
        if key in result:raise ValueError('duplicate JSON key')
        result[key]=value
    return result
def canonical(value):
    if not isinstance(value,str) or not value.startswith('/') or len(value)>4096 or '\\' in value or '..' in value.split('/') or any(ord(c)<32 or 127<=ord(c)<=159 for c in value):raise ValueError('unsafe path')
    return '/'+str(pathlib.PurePosixPath(value)).lstrip('/')
def under(path,root):
    path,root=pathlib.PurePosixPath(path),pathlib.PurePosixPath(root)
    return path==root or root in path.parents
FLAGS=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|getattr(os,'O_NONBLOCK',0)
def open_dir(path):
    fd=os.open('/',FLAGS)
    try:
        for part in pathlib.PurePosixPath(path).parts[1:]:
            following=os.open(part,FLAGS,dir_fd=fd);os.close(fd);fd=following
        return fd
    except BaseException:os.close(fd);raise
class ReadError(Exception):
    def __init__(self,code):self.code=code
def read(path,maximum,allow_directory=False):
    parent=open_dir(str(pathlib.PurePosixPath(path).parent));name=pathlib.PurePosixPath(path).name
    try:
        before=os.stat(name,dir_fd=parent,follow_symlinks=False)
        if allow_directory and stat.S_ISDIR(before.st_mode):
            following=os.open(name,FLAGS,dir_fd=parent);os.close(parent);parent=following
            name='norm_stats.json';path=str(pathlib.PurePosixPath(path)/name)
            before=os.stat(name,dir_fd=parent,follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):raise ReadError('DATASET_STATISTICS_UNSAFE_FILE')
        if not 0<before.st_size<=maximum:raise ReadError('DATASET_STATISTICS_SIZE_LIMIT')
        fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|getattr(os,'O_NONBLOCK',0),dir_fd=parent)
        try:
            initial=os.fstat(fd)
            if not stat.S_ISREG(initial.st_mode) or (initial.st_dev,initial.st_ino)!=(before.st_dev,before.st_ino):raise ReadError('DATASET_STATISTICS_SOURCE_CHANGED')
            chunks=[];size=0
            while size<=maximum:
                chunk=os.read(fd,min(65536,maximum+1-size))
                if not chunk:break
                size+=len(chunk);chunks.append(chunk)
            if size>maximum:raise ReadError('DATASET_STATISTICS_SIZE_LIMIT')
            after=os.fstat(fd);named=os.stat(name,dir_fd=parent,follow_symlinks=False)
            identity=lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
            if identity(initial)!=identity(after) or identity(initial)!=identity(named):raise ReadError('DATASET_STATISTICS_SOURCE_CHANGED')
            data=b''.join(chunks)
            document=json.loads(data,object_pairs_hook=pairs,parse_constant=reject)
            if not isinstance(document,dict):raise ValueError('JSON object required')
            return {'status':'READ','path':path,'sha256':hashlib.sha256(data).hexdigest(),'document':document}
        finally:os.close(fd)
    finally:os.close(parent)
def section(path,maximum,allow_directory=False):
    try:return read(path,maximum,allow_directory)
    except ReadError as error:code=error.code
    except FileNotFoundError:code='DATASET_STATISTICS_NOT_FOUND'
    except PermissionError:code='DATASET_STATISTICS_PERMISSION_DENIED'
    except OSError:code='DATASET_STATISTICS_UNSAFE_FILE'
    except (ValueError,TypeError,OverflowError,RecursionError,UnicodeError):code='DATASET_STATISTICS_INVALID_JSON'
    return {'status':'FAILED','path':path,'error':{'code':code}}
try:
    length=int(sys.stdin.buffer.readline(32))
    if not 0<length<=65536:raise ValueError('request size')
    payload=json.loads(sys.stdin.buffer.read(length));threading.Thread(target=disconnected,daemon=True).start()
    roots=[canonical(root) for root in payload['roots']]
    dataset=canonical(payload['dataset_path']);norm=canonical(payload['norm_path']) if payload.get('norm_path') else None
    if not roots or any(not any(under(path,root) for root in roots) for path in (dataset,norm) if path):raise ReadError('DATASET_STATISTICS_PATH_OUTSIDE_ROOTS')
    result={'dataset':section(str(pathlib.PurePosixPath(dataset)/'meta/info.json'),1024*1024),
        'normalization':section(norm,8*1024*1024,True) if norm else {'status':'UNBOUND','features':[]}}
    print(json.dumps({'result':result},ensure_ascii=False,allow_nan=False,separators=(',',':')))
except ReadError as error:print(json.dumps({'error':{'code':error.code,'message':'Cannot read registered metadata outside managed roots'}}))
except (ValueError,KeyError,TypeError,OverflowError,RecursionError,UnicodeError):print(json.dumps({'error':{'code':'DATASET_STATISTICS_INVALID_REQUEST','message':'Invalid bounded metadata request'}}))
'''

ERROR_MESSAGES = {
    'DATASET_STATISTICS_NOT_FOUND': '未找到已登记路径中的统计文件。',
    'DATASET_STATISTICS_PERMISSION_DENIED': 'SSH 用户无权读取该统计文件。',
    'DATASET_STATISTICS_UNSAFE_FILE': '统计文件包含链接、特殊文件或不可安全访问的目录。',
    'DATASET_STATISTICS_SIZE_LIMIT': '统计文件超过读取大小限制。',
    'DATASET_STATISTICS_SOURCE_CHANGED': '统计文件在读取期间发生变化，请重试。',
    'DATASET_STATISTICS_INVALID_JSON': '统计文件不是有效且有限的 JSON 对象。',
    'DATASET_STATISTICS_INVALID_METADATA': '数据集元信息字段、形状或数值无效。',
    'DATASET_STATISTICS_INVALID_NORM': '归一化统计各字段需包含等长、有限的 mean/std 等数值数组。',
    'DATASET_STATISTICS_INVALID_RESULT': '机器返回的统计结果无效。',
    'DATASET_STATISTICS_PATH_OUTSIDE_ROOTS': '已登记统计路径不在机器允许访问的目录内。',
    'DATASET_STATISTICS_TIMEOUT': '读取统计文件超时，请稍后重试。',
    'DATASET_STATISTICS_READ_FAILED': '无法读取统计文件，请检查 SSH 连接和文件权限。',
}


def public_error(code):
    if not isinstance(code, str) or code not in ERROR_MESSAGES:
        code = 'DATASET_STATISTICS_READ_FAILED'
    return {'code': code, 'message': ERROR_MESSAGES[code]}


def _text(value, maximum):
    if (not isinstance(value, str) or not 0 < len(value) <= maximum
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)):
        return False
    try:
        value.encode('utf-8')
        return True
    except UnicodeError:
        return False


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def normalize_info(document):
    if not isinstance(document, dict):
        raise ValueError('info must be an object')
    result = {'features': []}
    for key in ('total_frames', 'total_episodes'):
        if key in document:
            value = document[key]
            if type(value) is not int or not 0 <= value <= 2**63 - 1:
                raise ValueError('invalid dataset counts')
            result[key] = value
    if 'fps' in document:
        fps = document['fps']
        if not _finite(fps) or not 0 < fps <= 1000:
            raise ValueError('invalid metadata fps')
        result['fps'] = fps
    features = document.get('features')
    if not isinstance(features, dict) or not 1 <= len(features) <= 256:
        raise ValueError('invalid features')
    for key, feature in features.items():
        if not _text(key, 256) or not isinstance(feature, dict) or not _text(feature.get('dtype'), 64):
            raise ValueError('invalid feature name or dtype')
        shape = feature.get('shape')
        if (not isinstance(shape, list) or len(shape) > 8
                or any(type(value) is not int or not 0 < value <= 2**31 - 1 for value in shape)):
            raise ValueError('invalid feature shape')
        result['features'].append({'key': key, 'dtype': feature['dtype'], 'shape': shape})
    result['camera_count'] = sum(feature['dtype'].lower() in {'video', 'image'}
                                 for feature in result['features'])
    return result


def normalize_norm(document):
    stats = document.get('norm_stats') if isinstance(document, dict) else None
    if not isinstance(stats, dict) or not 1 <= len(stats) <= 256:
        raise ValueError('norm_stats is required')
    features = []
    for key, source in stats.items():
        if (not _text(key, 256) or not isinstance(source, dict)
                or source.get('mean') is None or source.get('std') is None):
            raise ValueError('normalization mean and std are required')
        feature = {'key': key}
        for field in ('mean', 'std', 'q01', 'q99', 'min', 'max'):
            array = source.get(field)
            if array is None:
                continue
            if (not isinstance(array, list) or not 1 <= len(array) <= 4096
                    or any(not _finite(value) for value in array)
                    or field == 'std' and any(value < 0 for value in array)):
                raise ValueError('invalid normalization values')
            feature[field] = array
        dimension = len(feature['mean'])
        if any(len(value) != dimension for name, value in feature.items() if name != 'key'):
            raise ValueError('normalization dimensions differ')
        for low, high in (('q01', 'q99'), ('min', 'max')):
            if low in feature and high in feature and any(a > b for a, b in zip(feature[low], feature[high])):
                raise ValueError('normalization interval is reversed')
        feature['dimension'] = dimension
        features.append(feature)
    return {'features': features}


def _section(value, path, kind):
    invalid_code = 'DATASET_STATISTICS_INVALID_METADATA' if kind == 'dataset' else 'DATASET_STATISTICS_INVALID_NORM'
    if not isinstance(value, dict):
        value = {}
    actual_path = value.get('path')
    allowed_paths = {path} if kind == 'dataset' else {path, str(PurePosixPath(path) / 'norm_stats.json')}
    if not isinstance(actual_path, str) or actual_path not in allowed_paths:
        return {'status': 'FAILED', 'path': path, 'features': [],
                'error': public_error('DATASET_STATISTICS_INVALID_RESULT')}
    if value.get('status') != 'READ':
        code = value.get('error', {}).get('code') if isinstance(value.get('error'), dict) else None
        return {'status': 'FAILED', 'path': actual_path, 'features': [], 'error': public_error(code)}
    try:
        sha = value.get('sha256')
        if not isinstance(sha, str) or len(sha) != 64 or any(char not in '0123456789abcdef' for char in sha):
            raise ValueError('invalid file digest')
        normalized = normalize_info(value.get('document')) if kind == 'dataset' else normalize_norm(value.get('document'))
        return {'status': 'READY', 'path': actual_path, 'source_sha256': sha, **normalized}
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError, UnicodeError):
        return {'status': 'FAILED', 'path': actual_path, 'features': [], 'error': public_error(invalid_code)}


def read_dataset_statistics(remote, dataset_path, norm_path, roots, *, timeout=8):
    """Read only info.json and the bound norm JSON; never load samples or tensors."""
    try:
        dataset_path = _path(dataset_path)
        norm_path = _path(norm_path) if norm_path is not None else None
        if not isinstance(roots, (list, tuple)) or not 1 <= len(roots) <= 100:
            raise ValueError('invalid roots')
        roots = [_path(root) for root in roots]
        if any(not any(_under(path, root) for root in roots) for path in (dataset_path, norm_path) if path):
            raise ValueError('outside roots')
    except (TransferError, ValueError, TypeError):
        raise TransferError('DATASET_STATISTICS_PATH_OUTSIDE_ROOTS', ERROR_MESSAGES['DATASET_STATISTICS_PATH_OUTSIDE_ROOTS']) from None
    payload = {'dataset_path': dataset_path, 'norm_path': norm_path, 'roots': roots}
    if not _finite(timeout) or not 0 < timeout <= 15:
        raise ValueError('Statistics timeout must be finite and at most 15 seconds')
    if len(json.dumps(payload).encode()) > 65536:
        raise TransferError('DATASET_STATISTICS_PATH_OUTSIDE_ROOTS', ERROR_MESSAGES['DATASET_STATISTICS_PATH_OUTSIDE_ROOTS'])
    try:
        result = _remote_json(remote, payload, lambda: False, timeout=timeout,
                              script=_READ_SCRIPT, max_bytes=MAX_RESULT_BYTES)
    except TransferError as error:
        code = 'DATASET_STATISTICS_TIMEOUT' if error.code == 'TRANSFER_TIMEOUT' else error.code
        clean = public_error(code)
        raise TransferError(clean['code'], clean['message']) from None
    return {'dataset': _section(result.get('dataset'), str(PurePosixPath(dataset_path) / 'meta/info.json'), 'dataset'),
            'normalization': _section(result.get('normalization'), norm_path, 'normalization') if norm_path else {'status': 'UNBOUND', 'features': []}}
