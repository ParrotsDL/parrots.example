## 前置工作

#### 安装必要的工具包
```
pip install -r requirements.txt --user
```

## 生成文档(建议)
```
  cd parrots.example/docs
  make html
```

## 自定义文档

若是不采用如上的框架，则可以采用这种方法来自定义文档框架。

#### step1. 新建文件夹存放文档，例如: docs
```
mkdir docs
cd docs
```

#### step2. 采用 sphinx 工具生成文档框架
```
 sphinx-quickstart --ext-autodoc  #配置相应的项目信息
```

#### step3. 进行文档项目配置
文档项目配置文件位于 `docs/source/conf.py`
```
cd docs/source
vim conf.py
```
在 `conf.py` 去注释以下三行 添加导向项目代码的路径
```
#import os
#import sys
#sys.path.insert(0, os.path.abspath('./../../'))
```
若想改变文档风格，可对 conf.py 进行配置（optional)
```
html_theme = 'alabaster' -> html_theme = 'sphinx_rtd_theme'
```

#### step4.编译文档
```
sphinx-apidoc -o ./source  ../
make html
```

说明：执行完以上步骤之后，生成的静态html页面位于目录 `docs/build/html`。

参考:[sphinx 文档](http://www.sphinx-doc.org/en/master/contents.html '参考文档')
