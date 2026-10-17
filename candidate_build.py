"""Round92 standalone allowlisted PEP517 builder; runtime bytes are unchanged.
New distribution tooling under the repository Apache-2.0 license.
"""
import base64,csv,gzip,hashlib,io,json,tarfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
NAME='tiger_memory'
VERSION='2.2.0a5+candidate92'
IDENTITY=NAME+'-'+VERSION

def _files():
    manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_bytes())
    files={}
    for name,pin in manifest['files'].items():
        p=ROOT/name
        if not p.is_file() or p.is_symlink() or any(x.is_symlink() for x in p.parents):raise ValueError('unsafe-source-path')
        raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('source-pin-drift')
        files[name]=raw
    files['SOURCE_MANIFEST.json']=(ROOT/'SOURCE_MANIFEST.json').read_bytes()
    return files

def _metadata(files):
    meta=('Metadata-Version: 2.1\nName: '+NAME+'\nVersion: '+VERSION+'\nSummary: Unadopted local v4 P3a candidate\nRequires-Python: >=3.9\nLicense: Apache-2.0\nDescription-Content-Type: text/markdown\n\n').encode()+files['README.md']
    return {'METADATA':meta,'WHEEL':b'Wheel-Version: 1.0\nGenerator: candidate92-allowlist\nRoot-Is-Purelib: true\nTag: py3-none-any\n','entry_points.txt':b'[console_scripts]\ntiger_memory = tiger_memory.cli:main\n','LICENSE':files['LICENSE'],'NOTICE':files['NOTICE'],'THIRD_PARTY.md':files['THIRD_PARTY.md']}

def get_requires_for_build_wheel(config_settings=None):return []
def get_requires_for_build_sdist(config_settings=None):return []
def prepare_metadata_for_build_wheel(metadata_directory,config_settings=None):
    d=Path(metadata_directory)/(IDENTITY+'.dist-info');d.mkdir(parents=True,exist_ok=True)
    for n,v in _metadata(_files()).items():(d/n).write_bytes(v)
    return d.name

def build_wheel(wheel_directory,config_settings=None,metadata_directory=None):
    src=_files();files={}
    for n,v in src.items():files[n if n.startswith('tiger_memory/') else 'tiger_memory/_bundle/'+n]=v
    prefix=IDENTITY+'.dist-info/';meta=_metadata(src)
    if metadata_directory is not None:
        for n,v in meta.items():
            if (Path(metadata_directory)/n).read_bytes()!=v:raise ValueError('metadata-drift')
    files.update({prefix+n:v for n,v in meta.items()})
    record=io.StringIO(newline='');writer=csv.writer(record,lineterminator='\n')
    for n,v in sorted(files.items()):writer.writerow([n,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(v).digest()).decode().rstrip('='),str(len(v))])
    writer.writerow([prefix+'RECORD','','']);files[prefix+'RECORD']=record.getvalue().encode()
    out=Path(wheel_directory);out.mkdir(parents=True,exist_ok=True);name=IDENTITY+'-py3-none-any.whl'
    with zipfile.ZipFile(out/name,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for n,v in sorted(files.items()):
            info=zipfile.ZipInfo(n,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.create_system=3;info.external_attr=0o100644<<16;z.writestr(info,v)
    return name

def build_sdist(sdist_directory,config_settings=None):
    files=_files();files['PKG-INFO']=_metadata(files)['METADATA'];out=Path(sdist_directory);out.mkdir(parents=True,exist_ok=True);name=IDENTITY+'.tar.gz'
    with (out/name).open('wb') as stream:
        with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as gz:
            with tarfile.open(fileobj=gz,mode='w',format=tarfile.PAX_FORMAT) as tar:
                for n,v in sorted(files.items()):
                    info=tarfile.TarInfo(IDENTITY+'/'+n);info.mode=0o644;info.mtime=0;info.size=len(v);tar.addfile(info,io.BytesIO(v))
    return name
