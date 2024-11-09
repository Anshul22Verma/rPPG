import pickle
import os
import hashlib
from functools import wraps

from config import config
from utils.helper import get_config_keys

cache_directory = get_config_keys(config, 'cache_dir')


def get_cache_filename(func, *args, **kwargs):
    """Generate a unique cache filename based on function name and parameters."""
    # Create a unique string for the combination of function and args/kwargs
    cache_key = str(func.__name__) + str(args) + str(kwargs)
    # Hash the string to create a filename (in case the string is too long)
    hash_key = hashlib.sha256(cache_key.encode('utf-8')).hexdigest()
    # Use a simple directory structure
    return f"{cache_directory}/{func.__name__}/{hash_key}.pkl"


def rPPGcache(func):
    """Decorator to cache function results using pickle."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        cache_filename = get_cache_filename(func, *args, **kwargs)
        
        # Check if cache file exists
        if os.path.exists(cache_filename):
            with open(cache_filename, 'rb') as f:
                print(f"Loading result from cache: {cache_filename}")
                return pickle.load(f)
        
        # If cache file doesn't exist, compute and cache the result
        result = func(*args, **kwargs)
        
        # Create the cache directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.dirname(cache_filename)), exist_ok=True)
        os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
        
        with open(cache_filename, 'wb') as f:
            print(f"Caching result to: {cache_filename}")
            pickle.dump(result, f)
        
        return result
    
    return wrapper

# TEST: caching Example function to demonstrate caching
@rPPGcache
def expensive_function(x, y):
    print("Computing expensive function...")
    return x * y

if __name__ == "__main__":
    from datetime import datetime
    starttime = datetime.now()

    expensive_function(100e-7, 456e+4)
    print(f"time taken: {datetime.now() - starttime}")
    starttime = datetime.now()
    expensive_function(100e-8, 456e+4)
    print(f"time taken using caching: {datetime.now() - starttime}")
