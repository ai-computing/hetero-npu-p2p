"""llm-d shared-storage control-plane adapter. KV bytes never pass through HTTP.

Call engine callbacks via an owning process when using the PDMA backend.
This module does not pretend to implement NIXL or Mooncake.
"""
from dataclasses import dataclass
import hashlib,json,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer


class StageError(Exception):
    def __init__(self,status,message):super().__init__(message);self.status=status


def validate_request(body,model,context):
    allowed={'model','prompt','max_tokens','max_completion_tokens','temperature','stream','cache_hit_threshold'}
    if not isinstance(body,dict) or set(body)-allowed:raise StageError(400,'unsupported request fields')
    if body.get('model')!=model:raise StageError(400,'model mismatch')
    prompt=body.get('prompt')
    if not isinstance(prompt,list) or not prompt or any(type(x) is not int or not 0<=x<128256 for x in prompt):
        raise StageError(400,'adapter requires one explicit token-ID prompt (KONI vocabulary)')
    count=body.get('max_completion_tokens',body.get('max_tokens',32))
    if type(count) is not int or count<=0 or len(prompt)+count>context:raise StageError(400,'request exceeds context')
    if body.get('temperature',0)!=0 or body.get('stream',False) is not False:raise StageError(400,'only greedy non-streaming requests supported')
    if 'max_tokens' in body and 'max_completion_tokens' in body and body['max_tokens']!=body['max_completion_tokens']:
        raise StageError(400,'conflicting output limits')
    return prompt,count


@dataclass
class Entry:
    fingerprint: str
    created: float
    handle: object = None
    state: str = 'admitting'


class StageBroker:
    def __init__(self,model,prepare,decode,discard,*,capacity=2,ttl=60,context=512,event_sink=None):
        if capacity<1 or ttl<=0:raise ValueError('positive capacity and TTL required')
        self.model,self.prepare,self.decode,self.discard=model,prepare,decode,discard
        self.capacity,self.ttl,self.context=capacity,ttl,context
        self.entries={};self.lock=threading.RLock();self.event_sink=event_sink or (lambda row:None)

    def _event(self,stage,rid,**kwargs):
        self.event_sink(dict(stage=stage,request_id=rid,time=time.perf_counter(),**kwargs))

    def _expire(self):
        now=time.monotonic()
        for rid,e in list(self.entries.items()):
            if now-e.created>self.ttl and e.state=='ready':
                self.discard(e.handle);del self.entries[rid];self._event('expired',rid)

    def execute(self,stage,rid,body):
        if stage not in ('prefill','decode') or not isinstance(rid,str) or not rid or len(rid)>128:
            raise StageError(400,'valid stage and x-request-id required')
        prompt,count=validate_request(body,self.model,self.context)
        fingerprint=hashlib.sha256(json.dumps([self.model,prompt],separators=(',',':')).encode()).hexdigest()
        with self.lock:
            self._expire();entry=self.entries.get(rid)
            if entry and entry.fingerprint!=fingerprint:raise StageError(409,'request ID reused for another prompt/model')
            if stage=='prefill':
                if entry:
                    if entry.state=='ready':return {'id':rid,'object':'text_completion','choices':[{'text':'','index':0,'finish_reason':'stop'}]}
                    raise StageError(409,'request is not reusable')
                if len(self.entries)>=self.capacity:raise StageError(429,'KV admission capacity reached')
                entry=Entry(fingerprint,time.monotonic());self.entries[rid]=entry
                try:
                    entry.handle=self.prepare(rid,prompt);entry.state='ready';entry.created=time.monotonic()
                    self._event('prefill_ready',rid)
                except BaseException:
                    entry.state='poisoned';self._event('prefill_failed',rid);raise
                return {'id':rid,'object':'text_completion','choices':[{'text':'','index':0,'finish_reason':'stop'}]}
            if entry is None or entry.state!='ready':raise StageError(409,'matching prefill is required before decode')
            entry.state='decoding';self._event('decode_start',rid)
            try:
                response=self.decode(entry.handle,prompt,count)
                if not isinstance(response,dict) or not isinstance(response.get('choices'),list):raise RuntimeError('decoder returned invalid response')
                del self.entries[rid];self._event('decode_complete',rid)
                return response
            except BaseException:
                entry.state='poisoned';self._event('decode_failed',rid);raise


def serve_stage(broker,stage,port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,payload):
            data=json.dumps(payload).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path not in ('/health','/ready'):self.reply(404,{'error':'not found'});return
            self.reply(200,{'ready':True,'stage':stage})
        def do_POST(self):
            if self.path!='/v1/completions':self.reply(404,{'error':'only /v1/completions supported'});return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=1048576:raise StageError(413,'invalid body size')
                body=json.loads(self.rfile.read(length));rid=self.headers.get('x-request-id')
                response=broker.execute(stage,rid,body);self.reply(200,response)
            except StageError as e:self.reply(e.status,{'error':str(e)})
            except (ValueError,TypeError):self.reply(400,{'error':'invalid JSON request'})
            except Exception:self.reply(500,{'error':'engine failure; cache quarantined'})
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    return server,thread
