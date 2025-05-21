import requests
import os
from typing import List,Dict,Any
from functools import lru_cache
import json
from pathlib import Path
from datetime import datetime
import logging
from cachetools import cached,TTLCache


logger=logging.getLogger(__name__)

class FacebookAPI:
    def __init__(self):
        self.access_token = os.getenv("FACEBOOK_ACCESS_TOKEN")
        self.base_url = "https://graph.facebook.com/v12.0"
        self.cache_dir=Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)


        self.memory_cache = TTLCache(maxsize=100, ttl=3600)  # Cache for 1 hour
    

    def _get_cache_key(self,endpoint:str,params:dict)->str:
        return f"{endpoint}_{hash(frozenset(params.items()))}"
    

    @cached(cache=TTLCache(maxsize=100, ttl=3600))
    def get_posts(self,page_id:str)-> List[Dict[str,Any]]:
        cache_key=self._get_cache_key(f"posts_{page_id}",{})
        cache_file=self.cache_dir/ f"{cache_key}.json"



        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        

        if cache_file.exists():
            with open(cache_file,"r") as f:
                data=json.load(f)
                self.memory_cache[cache_key]=data
                return data
        
        url=f"{self.base_url}/{page_id}/posts?access_token={self.access_token}"
        try:
            response=requests.get(url)
            response.raise_for_status()
            data=response,json()

            with open(cache_file,"w")as f:
                json.dump(data,f)
            self.memory_cache[cache_key]=data
            return data
        except  Exception as e:
            logger.error(f"facebook api error: {str(e)}")

            raise 

        

    
