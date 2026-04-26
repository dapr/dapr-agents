import threading,sys
from datetime import datetime,timezone
from typing import Any,Dict
L={"DEBUG":10,"INFO":20,"WARNING":30,"ERROR":40,"CRITICAL":50}
class SL:
    def __init__(self,name="dapr.agents",level="INFO"): self.name=name; self.lv=L.get(level.upper(),20); self._c=threading.local()
    def bind(self,**k): setattr(self._c,"data",{**getattr(self._c,"data",{}),**k}); return self
    def clear(self): self._c.data={}
    def _log(self,lvl,msg,**kw):
        c=getattr(self._c,"data",{}); r={"t":datetime.now(timezone.utc).isoformat(),"l":lvl,"a":c.get("a",self.name),"m":msg,"c":{**c,**kw}}
        if L.get(lvl,20)>=self.lv:
            print(r) if L.get(lvl,20)<40 else print(r,file=sys.stderr)
    def d(self,m,**k):self._log("DEBUG",m,**k)
    def i(self,m,**k):self._log("INFO",m,**k)
    def w(self,m,**k):self._log("WARNING",m,**k)
    def e(self,m,**k):self._log("ERROR",m,**k)
    def c(self,m,**k):self._log("CRITICAL",m,**k)
log=SL()