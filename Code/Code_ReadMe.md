#Code_ReadMe

## 0. Note
最新的代码相对原来的代码，做了很大的调整，但由于时间有限，没有进行运行和测试，也就是说代码可能有大量错误

##1.Sources
* Feats.py
* Models.py
* Solutions.py
* GetEnsWeight.py
* Submit.py
* MyUtil.py
* others: test.py, temp.py, back.py

##2.Function of source
### 2.1 Feats.py
* 对原始数据（input）进行预处理和特征工程，并将处理后的数据输出（feat）

### 2.2 Models.py
* 存放各种模型类：实现各个模型的运行过程（根据train和test，得到predict）

### 2.3 Soluitons.py
* Solution = Model + Feat
* 对solution进行参数调节（参数调节的过程记录在"log_write"），并将选择的参数输出('sol_sel')

### 2.4 GetEnsWeight.py
* Answer = solution + 相应的(param, cv_score, id,ensemble weight)
* 确定选择的solution的ensemble weight（将answers输出）

### 2.5 Submit.py
* 从answers的保存文件载入answers， 运行（bag）每个answer得到predict，并根据answer的weight进行融合

### 2.6 MyUtil.py
* 保存各源文件中会用到的常量和自己编写的一些功能函数

### 2.7 Others
* test: some paragram for test and check during some process  
* temp: temporary code
* back: useless code for now