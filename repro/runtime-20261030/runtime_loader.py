"""Load the fee-free service components before importing the stable adapter."""
from pathlib import Path
import importlib.util,sys
HERE=Path(__file__).resolve().parent

def load(name,filename):
    spec=importlib.util.spec_from_file_location(name,HERE/filename)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    import tiger_memory
    setattr(tiger_memory,name.rsplit('.',1)[-1],module)
    return module

def install():
    load('tiger_memory.embedding_checkpoint','embedding_checkpoint.py')
    load('tiger_memory.aml_add_lock','aml_add_lock.py')
    remote=load('tiger_memory.remote_embeddings','remote_embeddings.py')
    load('tiger_memory.database','database.py')
    load('tiger_memory.events','events.py')
    load('tiger_memory.search','search.py')
    load('tiger_memory.service','service.py')
    aml_adapter=load('tiger_memory.aml_adapter','aml_adapter.py')
    config=load('tiger_memory.aml_open_source','aml_open_source.py')
    return aml_adapter,remote,config
