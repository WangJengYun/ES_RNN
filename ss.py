import logging
 
# 基礎設定
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
 
# 定義 handler 輸出 sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# 設定輸出格式
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# handler 設定輸出格式
console.setFormatter(formatter)
# 加入 hander 到 root logger
logging.getLogger('').addHandler(console)
 
# root 輸出
logging.info('道可道非常道')
 
# 定義另兩個 logger
AA = logging.getLogger('myapp.area1')
B = loggng.getLogger('myapp.area2')
 
B.debug('天高地遠')
B.info('天龍地虎')
AA.warning('天發殺機')