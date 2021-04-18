# PHMLBE
Persistent topology based machine learning prediction of cluster binding energies
本人将于2021年7月1日毕业，之后将不再使用校园邮箱1801213239@pku.edu.cn，如果需要与本人连续，可以通过632746071@qq.com或微信chenxinaiguo。
本代码适用于vasp软件计算结果的特征向量提取，也可用与其他计算软件结果，不过需要稍微对代码进行更改。
# 如果直接用于vasp软件计算结果，需要注意以下几点：
  （1）readposcar函数用于读取poscar文件里的结构和原子数信息，输出是晶格参数信息和原子坐标([lattice,pos_all])。
  （2）getsuperpoint是用于生成超胞，超胞用于粒子群搜索算法的初代结构切割，如从超胞中切出10个原子的所有可能排布（最密堆积）。
  （3）makeposcar和point2poscar函数是readposcar的反函数，用于生成poscar文件。
  （4）search_close_point、ini_pos函数和random_point_gen函数用于生成粒子群算法的初代随机结构。
  （5）make_vasp函数用于生成vasp计算路径和计算文件。
  （6）if_cal_done_vasp用于判断vasp计算是否结束。
  （7）analysis_vasp用于提取vasp的计算结果——结构和能量，结构信息在poscar和XDATCAR里，poscar包含起始结构，XDATCAR是包含计算优化迭代时每一迭代步的结构，能量信息在vasp计算的输出vasp.out文件里。
  （8）calc_bonding_energy函数用于计算原子结合能。
  （9）pos2betti函数用于把点群数据转成betti特征向量，需要注意的是，点群数据是指在团簇的所有原子在1埃（A）坐标系下的原子坐标集合。如[[0,0,0],[1,0,0]]表示一个两个原子构成的团簇，其原子坐标分别是（0，0，0）和（1，0，0），两个原子相距1A。
  （10）get_train_model函数用于训练模型，如果是对vasp计算的poscar格式文件，直接使用即可，不过需要注意一下文件所在地址和读取顺序。target_list是读取文件夹列表，rank_list1是读取几代（粒子群）数据，
# 如果使用文章SI所提及格式的数据，需要注意：
  （1）在get_train_model中，有一个段pos_tmp,energy_tmp,mag_tmp=analysis_vasp(ji,atom_num)需要注意。读者可以删掉这部分，直接从data里面导入结构数据作为pos_tmp，能量数据作为energy_tmp，mag_tmp则完全复制energy_tmp或对应长度的[0,0,...,0]向量即可，mag_tmp本来设计用于磁信息提取和分析，后面没使用上。
  具体可以如下，把以下代码从get_train_model函数删去：\n
    for file_tmp in target_list:          
        atom_num=int(file_tmp )
        feature_train1=[]
        target_train1=[]
        os.chdir(file_tmp)
        rank_list=list(range(int(rank_list1[0]),int(rank_list1[1])+1))
        print(rank_list)
        for ji in rank_list:
            ji=str(ji)
            print(ji)
            if os.path.exists(str(ji)):
                pos_tmp,energy_tmp,mag_tmp=analysis_vasp(ji,atom_num)
            else:
                continue
            betti_tmp=pos2betti(pos_tmp)
            print(betti_tmp[0][0])
            far_tmp=pos_farest_dis(pos_tmp)
            energy_tmp=calc_bonding_energy(energy_tmp,atom_num,if_even=1)
           for j3 in range(len(betti_tmp)):
                betti_tmp[j3].append(round(far_tmp[j3],2))
                betti_tmp[j3].append(round(far_tmp[j3]*far_k)/far_k)
            feature_train1=feature_train1+betti_tmp
            if label=='energy':
                target_train1=target_train1+energy_tmp
        mag_tmp=list(range(len(target_train1)))
        feature_train1,target_train1,mag_tmp=betti_clean(feature_train1,target_train1,mag_tmp)
        feature_train=feature_train+feature_train1
        target_train=target_train+target_train1
        os.chdir(work_file)
        print(len(target_train))
        print('\\\\\\\\\\\\\\\\\\')
        print(len(feature_train))
 
 这一段是用于从vasp计算结果生成feature_train和target_train的。
 在你的代码里，应如下。
 其中feature_train为TheDataOfClusters.py中提取出结构数据集合，举个简单例子，如[[[[1,3,3],[2,3,54],[3,34,5]],[[1,2,33],[2,3,4],[3,4,5]],[[1,2,3],[2,3,42],[3,4,5]]],[[[0,0,0],[2,3,4],[3,4,5]],[[0,0,0],[2,4,4],[3,4,5]],[[8,2,3],[2,3,4],[3,4,5]]]]，其中包含两个团簇，每个团簇包含三个原子。
 其中target_train1为能量数据，如[3.425,6.212]。大概意思如此，具体实现时还需充分发挥你的聪明才智，以及需要注意numpy的使用可能带来的格式问题。
