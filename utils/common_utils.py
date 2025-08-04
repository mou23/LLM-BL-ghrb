import json
from scipy import sparse
import pickle
import time
from contextlib import contextmanager
import numpy as np

@contextmanager
def timing(task=""):
    print(f"{task} started!")
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{task} runing time: {elapsed_time/60:.2f} min")
    
def save_file(path, data):
    assert path.endswith('pkl') or path.endswith('json') or path.endswith('npy') or path.endswith('npz') or path.endswith('txt')
    if path.endswith('pkl'):
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    elif path.endswith('json') or path.endswith('txt'):
        with open(path,'w') as f:
            json.dump(data,f)
    elif path.endswith('npy'):
        np.save(path,data)
    elif path.endswith('npz'):
        sparse.save_npz(path,data)

        
def load_file(path):
    assert path.endswith('pkl') or path.endswith('json') or path.endswith('npy') or path.endswith('npz') or path.endswith('txt')
    if path.endswith('json') or path.endswith('txt'):
        with open(path,'r') as f:
            data = json.load(f)
        return data
    elif path.endswith('pkl'):
        with open(path,'rb') as f:
            data = pickle.load(f)
        return data
    elif path.endswith('npy'):
        data = np.load(path)
        return data
    elif path.endswith('npz'):
        data = sparse.load_npz(path)
        return data

def get_rel_file(input_string):
    # index = input_string.find("org/apache")
    # result_string = input_string[index:] if index != -1 else input_string
    # result_string = result_string.replace("/",".")
    return input_string.replace("\\","/")
   

class DiffTimer(object):
    last = None
    
    def __init__(self):
        self.set()
        pass
    
    def set(self):
        self.last = time.time()
        pass
    
    def diff(self):
        diff = time.time() - self.last
        self.set()
        return diff
    
    def diff_millisecond(self):
        diff = int(self.diff()*1000)
        return u'%dms' % diff
    
    def diff_seconds(self):
        diff = int(self.diff()*1000)
        return u'%ds %dms' % (diff/1000, diff%1000)
        
    def diff_minute(self):
        diff = int(self.diff())
        return u'%dm %ds' % (diff/60, diff%60)
    
    def diff_hour(self):
        diff = int(self.diff())
        m = diff/60
        return u'%dh %dm %ds' % (m/60, m%60, diff%60)
    
    def diff_day(self):
        diff = int(self.diff())
        m = diff/60
        h = m/60
        return u'%dd %dh %dm' % (h/24, h%24, m)

    def diff_auto(self):
        diff = int(self.diff()*1000)
        ms = diff%1000  #get millisecond
        s = diff/1000
        m = s/60        #get minute
        s = s%60
        h = m/60
        m = m%60
        d = h/24
        h = h%24

        text = u''
        if d!=0:
            text += u'%dd '%d
        if h!=0 or text!=u'':
            text += u'%02d:'%h
        if m!=0 or text!=u'':
            text += u'%02d:'%m
        if s!=0 or text!=u'':
            text += u'%02d.'%s
        if ms!=0 or text!=u'':
            text += u'%03d.'%ms
        text = text[:-1]

        if len(text)==0:
            text =u'0ms'
        elif len(text)<2:
            text += u'ms'
        return text

class Progress(object):
    header = u''
    dot_point = 10
    line_point = 1000
    point = 0
    upper_bound = 0

    prev = 0
    percent_mode = False

    timer = None
    fulltimer = None

    def __init__(self, _header, _dot_point=0, _line_point=0, _percent_mode=False):
        self.reset(_header, _dot_point, _line_point, _percent_mode)

    def reset(self, _header, _dot_point=0, _line_point=0, _percent_mode=False):
        self.header = _header
        self.set_dotpoint(_dot_point)
        self.set_linepoint(_line_point)
        self.point = 0
        self.timer = DiffTimer()
        self.fulltimer = DiffTimer()
        self.prev = 1
        self.percent_mode = _percent_mode

    def set_header(self, _header):
        self.header = _header
        return self

    def set_upperbound(self, _max):
        self.upper_bound = _max
        return self

    def set_point(self, _point):
        self.point = _point
        return self

    def set_dotpoint(self, _dot_point):
        self.dot_point = _dot_point if _dot_point > 0 else 1
        return self

    def set_linepoint(self, _line_point):
        self.line_point = _line_point if _line_point > 0 else 1
        return self

    def start(self):
        print(u'%s'%self.header, end=u'')
        self.prev = 1
        self.timer.set()
        self.fulltimer.set()


    def _percent(self):

        div = int ((float(self.point) / self.upper_bound) * 100)
        if div >= 100: div=100

        for i in range(self.prev,div):
            if i==0: continue
            elif i%self.line_point==0: print(u',', end=u'')
            elif i%self.dot_point==0:print(u'.', end=u'')

        self.prev = div

    def check(self, _msg=None):
        self.point += 1

        # work with percent
        if self.percent_mode is True:
            self._percent()
            return

        if (self.point % self.dot_point) == 0:
            print(u'.', end=u'')

        if (self.point % self.line_point) == 0:
            text = u'%s'%(u'{:,}'.format(self.point))
            if self.upper_bound >0:
                text += u'/%s'%(u'{:,}'.format(self.upper_bound))
            text += u' (time:%s'%self.timer.diff_auto()
            if _msg is not None:
                text += u' %s' %_msg
            text += u')'
            print(text)
            print(u'%s'%self.header, end=u'')

    def done(self, _msg=None):
        text = u'Done. (size:%s'%(u'{:,}'.format(self.point))
        text += u' time:%s'% self.fulltimer.diff_auto()
        if _msg is not None:
            text += u' %s)'%_msg
        else:
            text += u')'
        print(text)

class PrettyStringBuilder(object):
    accuracy = 2
    indent_depth = 3
    point_level = 0

    def __init__(self, _indent_depth=3, _accuracy=2, _point_level=0):
        self.accuracy = _accuracy
        self.indent_depth = _indent_depth
        self.point_level = _point_level
        pass

    def toString(self, _item):
        return self.get_itemtext(_item)

    def get_itemtext(self, _item, _indent=0):
        if isinstance(_item, dict):
            return self.get_dicttext(_item, _indent)
        elif isinstance(_item, list):
            return self.get_listtext(_item, _indent)
        elif isinstance(_item, set):
            return self.get_listtext(_item, _indent)
        elif isinstance(_item, str):
            return u'"%s"' % self.escape(_item)
        elif isinstance(_item, float):
            return str(_item) if self.accuracy <= 0 else ((u'%%.%df' % (self.accuracy)) % _item)
        elif isinstance(_item, int):
            return str(_item) if self.point_level <= 0 else self.get_integer(_item)
        else:
            return u'%s' % str(_item)

    def get_integer(self, _value):
        text = str(_value)
        count = 0
        position = self.point_level + count
        while True:
            if len(text) > position:
                text = text[:-position] + u',' + text[-position:]
                count+=1
                position += self.point_level + count
            else:
                break
        return text

    def get_listtext(self, _items, _indent=0):
        '''
        make list text
        :param _items:
        :param _indent:
        :return:
        '''
        if len(_items) == 0: return u'[]'

        # make list opener
        text = u'['

        # make dict items text
        for value in _items:
            text += u'%s%s, ' % (
                u'' if _indent >= self.indent_depth else (u'\n' + u'\t' * (_indent + 1)),
                self.get_itemtext(value, _indent + 1)
            )

        # make dict closer
        text = text.strip()
        if text.endswith(','): text = text[:-1]
        text += u'%s]' % (u'' if _indent >= self.indent_depth else (u'\n' + u'\t' * _indent))

        return text

    def get_dicttext(self, _item, _indent=0):
        '''
        make dictionary text
        :param _item:
        :param _indent:
        :return:
        '''
        if len(_item) == 0:    return u'{}'

        # make dict opener
        text = u'{'

        # make dict items text
        for key, value in _item.items():
            text += u'%s%s:%s, '% (
                u'' if _indent >= self.indent_depth else (u'\n' + u'\t' * (_indent+1)),
                self.get_keytext(key),
                self.get_itemtext(value, _indent+1)
            )

        # make dict closer
        text = text.strip()
        if text.endswith(','): text = text[:-1]
        text += u'%s}' % (u'' if _indent >= self.indent_depth else (u'\n' + u'\t'*_indent))

        return text

    def get_keytext(self, _key):
        '''
        return text for dictionary key
        :param _key:
        :return:
        '''
        if isinstance(_key, str):
            return u'\"%s\"' % _key
        elif isinstance(_key, int):
            return u'%d' % _key
        elif isinstance(_key, float):
            return u'%.f' % (_key if self.accuracy is None else (u'%%.%df' % self.accuracy) % _key)
        else:
            return u'\'%s\'' % _key.__hash__()

    def escape(self, text):
        return text.replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r').replace('\"', '\\"')
