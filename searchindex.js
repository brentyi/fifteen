Search.setIndex({docnames:["api/fifteen/data/_cycled_minibatches/index","api/fifteen/data/_dataloader/index","api/fifteen/data/_in_memory_dataloader/index","api/fifteen/data/_prefetching_map/index","api/fifteen/data/_protocols/index","api/fifteen/data/_sharding_map/index","api/fifteen/data/index","api/fifteen/experiments/_experiment/index","api/fifteen/experiments/_log_data/index","api/fifteen/experiments/index","api/fifteen/index","api/fifteen/utils/_git/index","api/fifteen/utils/_hcb_print/index","api/fifteen/utils/_loop_metrics/index","api/fifteen/utils/_pdb_safety_net/index","api/fifteen/utils/_stopwatch/index","api/fifteen/utils/_timestamp/index","api/fifteen/utils/index","index"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["api/fifteen/data/_cycled_minibatches/index.rst","api/fifteen/data/_dataloader/index.rst","api/fifteen/data/_in_memory_dataloader/index.rst","api/fifteen/data/_prefetching_map/index.rst","api/fifteen/data/_protocols/index.rst","api/fifteen/data/_sharding_map/index.rst","api/fifteen/data/index.rst","api/fifteen/experiments/_experiment/index.rst","api/fifteen/experiments/_log_data/index.rst","api/fifteen/experiments/index.rst","api/fifteen/index.rst","api/fifteen/utils/_git/index.rst","api/fifteen/utils/_hcb_print/index.rst","api/fifteen/utils/_loop_metrics/index.rst","api/fifteen/utils/_pdb_safety_net/index.rst","api/fifteen/utils/_stopwatch/index.rst","api/fifteen/utils/_timestamp/index.rst","api/fifteen/utils/index.rst","index.rst"],objects:{"":[[10,0,0,"-","fifteen"]],"fifteen.data":[[6,1,1,"","DataLoader"],[6,1,1,"","DataLoaderProtocol"],[6,1,1,"","InMemoryDataLoader"],[6,1,1,"","MapDatasetProtocol"],[6,1,1,"","SizedIterable"],[0,0,0,"-","_cycled_minibatches"],[1,0,0,"-","_dataloader"],[2,0,0,"-","_in_memory_dataloader"],[3,0,0,"-","_prefetching_map"],[4,0,0,"-","_protocols"],[5,0,0,"-","_sharding_map"],[6,5,1,"","cycled_minibatches"],[6,5,1,"","prefetching_map"],[6,5,1,"","sharding_map"]],"fifteen.data.DataLoader":[[6,2,1,"","__post_init__"],[6,3,1,"","collate_fn"],[6,3,1,"","dataset"],[6,3,1,"","drop_last"],[6,2,1,"","minibatch_count"],[6,3,1,"","minibatch_size"],[6,2,1,"","minibatches"],[6,3,1,"","num_workers"],[6,3,1,"","workers_state"]],"fifteen.data.DataLoaderProtocol":[[6,3,1,"","drop_last"],[6,2,1,"","minibatch_count"],[6,3,1,"","minibatch_size"],[6,2,1,"","minibatches"]],"fifteen.data.InMemoryDataLoader":[[6,2,1,"","__post_init__"],[6,3,1,"","dataset"],[6,3,1,"","drop_last"],[6,2,1,"","minibatch_count"],[6,3,1,"","minibatch_size"],[6,2,1,"","minibatches"],[6,3,1,"","sample_count"]],"fifteen.data.MapDatasetProtocol":[[6,2,1,"","__getitem__"],[6,2,1,"","__len__"]],"fifteen.data._cycled_minibatches":[[0,4,1,"","PyTreeType"],[0,5,1,"","cycled_minibatches"]],"fifteen.data._dataloader":[[1,4,1,"","CollateFunction"],[1,1,1,"","DataLoader"],[1,4,1,"","PyTreeType"],[1,4,1,"","T"]],"fifteen.data._dataloader.DataLoader":[[1,2,1,"","__post_init__"],[1,3,1,"","collate_fn"],[1,3,1,"","dataset"],[1,3,1,"","drop_last"],[1,2,1,"","minibatch_count"],[1,3,1,"","minibatch_size"],[1,2,1,"","minibatches"],[1,3,1,"","num_workers"],[1,3,1,"","workers_state"]],"fifteen.data._in_memory_dataloader":[[2,1,1,"","InMemoryDataLoader"],[2,4,1,"","PyTreeType"]],"fifteen.data._in_memory_dataloader.InMemoryDataLoader":[[2,2,1,"","__post_init__"],[2,3,1,"","dataset"],[2,3,1,"","drop_last"],[2,2,1,"","minibatch_count"],[2,3,1,"","minibatch_size"],[2,2,1,"","minibatches"],[2,3,1,"","sample_count"]],"fifteen.data._prefetching_map":[[3,4,1,"","PyTreeType"],[3,5,1,"","prefetching_map"]],"fifteen.data._protocols":[[4,4,1,"","ContainedType"],[4,1,1,"","DataLoaderProtocol"],[4,1,1,"","MapDatasetProtocol"],[4,1,1,"","SizedIterable"]],"fifteen.data._protocols.DataLoaderProtocol":[[4,3,1,"","drop_last"],[4,2,1,"","minibatch_count"],[4,3,1,"","minibatch_size"],[4,2,1,"","minibatches"]],"fifteen.data._protocols.MapDatasetProtocol":[[4,2,1,"","__getitem__"],[4,2,1,"","__len__"]],"fifteen.data._sharding_map":[[5,4,1,"","PyTreeType"],[5,5,1,"","sharding_map"]],"fifteen.experiments":[[9,1,1,"","Experiment"],[9,1,1,"","TensorboardLogData"],[7,0,0,"-","_experiment"],[8,0,0,"-","_log_data"]],"fifteen.experiments.Experiment":[[9,2,1,"","assert_exists"],[9,2,1,"","assert_new"],[9,2,1,"","clear"],[9,3,1,"","data_dir"],[9,2,1,"","log"],[9,2,1,"","move"],[9,2,1,"","read_metadata"],[9,2,1,"","restore_checkpoint"],[9,2,1,"","save_checkpoint"],[9,2,1,"","summary_writer"],[9,3,1,"","verbose"],[9,2,1,"","write_metadata"]],"fifteen.experiments.TensorboardLogData":[[9,2,1,"","fix_sharded_scalars"],[9,3,1,"","histograms"],[9,2,1,"","merge"],[9,2,1,"","merge_histograms"],[9,2,1,"","merge_scalars"],[9,2,1,"","prefix"],[9,3,1,"","scalars"]],"fifteen.experiments._experiment":[[7,1,1,"","Experiment"],[7,4,1,"","Pytree"],[7,4,1,"","PytreeType"],[7,4,1,"","T"],[7,4,1,"","cached_property"]],"fifteen.experiments._experiment.Experiment":[[7,2,1,"","assert_exists"],[7,2,1,"","assert_new"],[7,2,1,"","clear"],[7,3,1,"","data_dir"],[7,2,1,"","log"],[7,2,1,"","move"],[7,2,1,"","read_metadata"],[7,2,1,"","restore_checkpoint"],[7,2,1,"","save_checkpoint"],[7,2,1,"","summary_writer"],[7,3,1,"","verbose"],[7,2,1,"","write_metadata"]],"fifteen.experiments._log_data":[[8,4,1,"","Array"],[8,4,1,"","Scalar"],[8,4,1,"","T"],[8,1,1,"","TensorboardLogData"]],"fifteen.experiments._log_data.TensorboardLogData":[[8,2,1,"","fix_sharded_scalars"],[8,3,1,"","histograms"],[8,2,1,"","merge"],[8,2,1,"","merge_histograms"],[8,2,1,"","merge_scalars"],[8,2,1,"","prefix"],[8,3,1,"","scalars"]],"fifteen.utils":[[17,1,1,"","LoopMetrics"],[11,0,0,"-","_git"],[12,0,0,"-","_hcb_print"],[13,0,0,"-","_loop_metrics"],[14,0,0,"-","_pdb_safety_net"],[15,0,0,"-","_stopwatch"],[16,0,0,"-","_timestamp"],[17,5,1,"","get_git_commit_hash"],[17,5,1,"","hcb_print"],[17,5,1,"","loop_metric_generator"],[17,5,1,"","pdb_safety_net"],[17,5,1,"","stopwatch"],[17,5,1,"","timestamp"]],"fifteen.utils.LoopMetrics":[[17,3,1,"","counter"],[17,3,1,"","iterations_per_sec"],[17,3,1,"","time_elapsed"]],"fifteen.utils._git":[[11,5,1,"","get_git_commit_hash"]],"fifteen.utils._hcb_print":[[12,5,1,"","hcb_print"]],"fifteen.utils._loop_metrics":[[13,1,1,"","LoopMetrics"],[13,5,1,"","loop_metric_generator"],[13,4,1,"","loop_metrics"],[13,5,1,"","range_with_metrics"]],"fifteen.utils._loop_metrics.LoopMetrics":[[13,3,1,"","counter"],[13,3,1,"","iterations_per_sec"],[13,3,1,"","time_elapsed"]],"fifteen.utils._pdb_safety_net":[[14,5,1,"","pdb_safety_net"]],"fifteen.utils._stopwatch":[[15,5,1,"","stopwatch"]],"fifteen.utils._timestamp":[[16,5,1,"","timestamp"]],fifteen:[[6,0,0,"-","data"],[9,0,0,"-","experiments"],[17,0,0,"-","utils"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","data","Python data"],"5":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:data","5":"py:function"},terms:{"0":[1,6,13,15,17],"05":[16,17],"1":[0,6,7,9,13,15,17],"11":[16,17],"15":[16,17],"2":[3,6],"2021":[16,17],"32":[16,17],"46":[16,17],"abstract":[0,6],"break":[8,9],"case":[3,6],"default":[1,6],"float":[13,17],"function":[1,7,8,9],"int":[0,1,2,3,4,6,7,9,13,17],"new":[7,9],"return":[1,2,6,7,8,9,11,15,17],"true":[1,2,6,7,9,13,17],"while":[13,17],By:[1,6],For:[3,6,7,9],If:[3,6],In:[8,9],It:[1,2,6],The:[2,6],There:[7,9],To:[8,9],__contains__:[4,6],__getitem__:[1,4,6],__iter__:[4,6],__len__:[1,4,6],__post_init__:[1,2,6],_collections_abc:[4,6],_workersst:[1,6],abc:[4,6],about:[0,6],access:[7,9],across:[8,9],adapt:[3,6],add:[8,9],all:[7,8,9],alloc:[3,6],along:[1,6],also:[8,9],amount:[1,2,6],an:[1,2,3,5,6,7,8,9,13,14,17,18],analysi:[8,9],ani:[7,9,12,17],approach:[8,9],approxim:[14,17],ar:[0,1,2,3,4,5,6,7,9],arbitrari:[1,6],arg:[12,17],arithmet:[8,9],around:[7,9],arrai:[1,2,3,6,8,9],assert_exist:[7,9],assert_new:[7,9],associ:[7,9],attach:[14,17],augment:[0,6],automat:[14,17],averag:[8,9],awai:[0,6],axi:[1,2,5,6],b:[14,17],base:[1,2,4,6],batch:[1,2,6],befor:[3,6],blob:7,block:[15,17],bool:[1,2,4,6,7,9],both:[3,4,6],brentyi:7,buffer:[3,6],buffer_s:[3,6],bypass:[14,17],c:[14,17],cached_properti:7,call:[8,9,14,17],callabl:[12,17],callback:[12,17],can:[0,1,2,3,4,6],checkpoint:[7,9],checkpoint_:[7,9],clear:[7,9],co:[7,9],code:[7,9,15,17],collat:[1,6],collate_fn:[1,6],collatefunct:[1,6],collect:[4,6],com:7,combin:[3,6],commit:[3,5,6,11,17],common:[7,9],compat:[8,9],compil:[1,2,6],comput:[1,2,6,13,17],concept:[0,6],concern:[1,2,6],construct:[1,2,6],contain:[8,9],containedtyp:[4,6],context:[15,17],continu:[14,17],correspond:[2,5,6,7,9],could:[8,9],count:[2,4,6],counter:[13,17],cpu:[3,6],creat:[8,9],ctrl:[14,17],current:[3,6,8,9,11,16,17],cwd:[11,17],cycled_minibatch:[0,6],data:[7,8,9,10,18],data_dir:[7,9],dataclass:[7,9],dataload:[0,1,4,6],dataloaderprotocol:[0,1,2,4,6],dataset:[0,1,2,4,6],dcarg:[7,9],decreas:[1,2,6],defin:[4,6,7,9],delet:[7,9],devic:[3,5,6,8,9],device_count:[5,6],device_put:[3,6],dfgo:7,diagnos:[14,17],dict:[8,9],dictionari:[8,9],differ:[0,6],directori:[7,9],disabl:[1,6],distribut:[5,6,8,9],divis:[1,2,6],doe:[4,6],doesn:[8,9],done:[8,9],drop:[1,2,6],drop_last:[1,2,4,6],due:[4,6],each:[0,2,5,6,8,9],either:[14,17],ellipsi:[12,17],emul:[14,17],encount:[14,17],epoch:[0,1,2,6],error:[14,17],evalu:[4,6],evenli:[1,2,6],everi:[2,6],exampl:[13,15,16,17],except:[14,17],exist:[7,9],expect:[1,6],expected_typ:[7,9],experi:[10,18],experiment_fil:7,explicit:[3,6],fals:[7,9],file:[7,9],finish:[3,6],first:[2,6,8,9,13,17],fit:[1,6],fix:[8,9],fix_sharded_scalar:[7,8,9],flatten:[8,9],flax:[3,6,7,9],folder:[8,9],format:[16,17],free:[3,6],friendli:[12,17],from:[1,2,3,6,7,8,9],futur:[8,9],gener:[1,2,3,4,6,13,15,17],get:[16,17],get_git_commit_hash:[11,17],git:[11,17],github:7,gpu:[3,6],gradient:[1,2,6],handl:[0,6,7,9],hash:[11,17],have:[1,2,6],hcb_print:[12,17],help:[14,17],helper:[7,9,12,15,17],here:[7,8,9],histogram:[8,9],hit:[14,17],hood:[0,3,6],host:[12,17],http:7,i:[2,6],ident:[3,6],implement:[1,4,6,7,9],implicit:[0,6],improv:[3,6],includ:[7,9],increment:[0,6],index:[1,2,4,5,6],indic:[1,6],infinit:[0,6,13,17],infrastructur:18,inmemorydataload:[2,6],input:[3,5,6],insid:[7,9],instanc:[7,9],instead:[7,9,13],integ:[1,6,13],involv:[0,6],item:[0,1,6],iter:[0,1,2,3,4,5,6],iteration_per_sec:[13,17],iterations_per_sec:[13,17],jax:[1,2,3,5,6,8,9,18],jax_util:[3,6],jit:[1,2,6,12,17],keep:[7,9],keep_every_n_step:[7,9],kwarg:[12,17],label:[15,17],lambda:[2,6],larg:[1,6],last:[1,2,6],launch:[14,17],lax:[8,9],lead:[5,6],leaf:[5,6],length:[4,6],lib:[3,5,6,7],librari:18,light:13,like:[4,6],littl:[7,9],live:[3,6],load:[4,6,7,9],loader:[1,2,6],locat:[7,9],log:[7,8,9],log_data:[7,9],log_histograms_every_n:[7,9],log_scalars_every_n:[7,9],logic:[0,6],longer:[8,9],loop:[3,6,13,17],loop_metr:[13,17],loop_metric_gener:[13,17],loopmetr:[13,17],m:[14,17],make:[3,6,7,8,9],manag:[4,6,7,15,17],mani:[8,9],map:[1,3,4,5,6],mapdatasetprotocol:[1,4,6],master:7,mean:[8,9],memori:[1,2,3,6],merg:[8,9],merge_histogram:[8,9],merge_scalar:[8,9],metadata:[7,9],method:[4,6],metric:[0,6,7,8,9,13,17],might:[8,9],minibatch:[1,2,3,4,6],minibatch_count:[1,2,4,6],minibatch_s:[1,2,4,6],minor:[14,17],mislead:[0,6],model:[14,17],move:[7,9],multi:[3,6],multipl:[5,6,8,9],multiprocess:[1,4,6],n:[5,6],name:[7,8,9,15,17],net:[14,17],new_data_dir:[7,9],next:[3,6,13,17],nice:[1,2,6],noisi:[1,2,6],none:[3,6,7,9,11,12,13,15,17],note:[13,17],num_work:[1,6],number:[1,2,4,6],object:[4,6,7,9,13],often:[0,6,8,9],onc:[3,6],one:[3,6],onli:[4,6],onto:[3,6],open:[14,17],optim:[4,6],option:[0,1,2,3,4,6,7,9,11,17],order:[4,6],other:[8,9],our:[1,2,6],output:[5,6],over:[0,1,2,3,4,5,6],overwrit:[7,9],parallel:[3,6],paramet:[15,17],particularli:[0,3,4,6],path:[7,9,11,17],pathlib:[7,9,11,17],pdb:[14,17],pdb_safety_net:[14,17],per:[1,2,6],perform:[8,9],place:[13,15,17],pmap:[7,8,9],pmean:[8,9],prefetch:[3,4,6],prefetch_to_devic:[3,6],prefetching_map:[3,6],prefix:[7,8,9],print:[12,13,15,17],problem:[14,17],progress:[0,6],properti:[7,9],protocol:[1,2,4,6],push:[3,6],py:[7,14,17],python:[14,17],pytorch:[1,4,6],pytre:[1,2,3,5,6,7,9],pytreetyp:[0,1,2,3,5,6,7,9],pyyaml:[7,9],random:[1,2,6],rang:13,range_with_metr:13,re:[3,6],read_metadata:[7,9],real:[7,9],reason:[0,6],reduc:[1,2,6],replac:[8,9],represent:[8,9],requir:[4,6],rescu:[14,17],restore_checkpoint:[7,9],run:[7,9,15,17],runtim:[15,17],s:[1,2,6,7,9,15,17],safeti:[14,17],same:[1,2,6],sampl:[2,4,6],sample_count:[2,6],save:[7,9],save_checkpoint:[7,9],scalar:[8,9],scope:[8,9],script:[7,9,14,17],script_nam:[14,17],see:[7,9],seed:[0,1,2,6],self:[1,2,4,6,7,8,9],sens:[8,9],sequenc:[2,5,6],serial:[7,9],set:[1,3,6],shape:[5,6],shard:[5,6,8,9],sharding_map:[3,5,6],should:[1,2,3,6],shuffl:[0,1,2,6],shuffle_se:[0,1,2,4,6],similar:[1,4,6],simpl:[2,6,7,9],simpli:[1,6,8,9],sinc:[8,9],singl:[4,6],size:[0,1,2,3,4,6],sizediter:[1,3,4,5,6],sleep:[13,15,17],small:[1,2,6],some:[7,8,9],sourc:[0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17],special:[7,9],specifi:[3,6],spent:[1,2,6],stack:[1,6],standard:[3,6,8,9],start:13,stateless:[1,6],statement:[15,17],step:[7,9,13],still:[3,6],stop:13,stopwatch:[15,17],store:[2,6],str:[7,8,9,11,12,15,16,17],string:[7,9,16,17],string_from_arg:[12,17],structur:[2,4,6,8,9],style:[4,6],summari:[7,9],summary_writ:[7,9],summarywrit:[7,9],supertyp:4,support:[8,9],sure:[7,9],t:[1,7,8,9],tag:[8,9],take:[5,6],target:[1,6,7,9],tensorboard:[7,8,9],tensorboardlogdata:[7,8,9],them:[3,6,8,9],therefor:[2,6],thi:[3,4,6,7,8,9,13,17],thin:[7,9],those:[8,9],time:[1,2,6,13,15,17],time_elaps:[13,17],timestamp:[16,17],too:[1,6],tool:[4,6],total:[2,4,6],tqdm:[4,6],train:[0,3,6,7,9],transform:[7,8,9],tree_map:[2,6],two:[3,4,6,8,9],type:[1,2,4,6,7,9],typic:[3,6],uncaught:[14,17],under:[0,3,6],unexpect:[14,17],unmodifi:[8,9],unsav:[14,17],updat:[7,9],us:[1,2,3,4,6,7,8,9,13],usag:[13,17],user:[14,17],usual:[1,2,6],util:[8,9,10,18],valu:[8,9],verbos:[7,9],veri:[1,2,6,7,9],via:[2,3,6,7,9,12,14,17],view:[8,9],we:[1,3,6,7,8,9,14,17],when:[0,3,6,8,9,14,17],where:[5,6,7,9],which:[0,1,4,5,6],within:[2,6],work:4,workers_st:[1,6],wrapper:[7,9,13],write_metadata:[7,9],writer:[7,9],x:[2,6],xla_client:[3,5,6],yaml:[7,9],yield:[3,6,13]},titles:["<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">fifteen.data._cycled_minibatches</span></code>","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">fifteen.data._dataloader</span></code>","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">fifteen.data._in_memory_dataloader</span></code>","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">fifteen.data._prefetching_map</span></code>","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">fifteen.data._protocols</span></code>","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">fifteen.data._sharding_map</span></code>","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">fifteen.data</span></code>","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">fifteen.experiments._experiment</span></code>","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">fifteen.experiments._log_data</span></code>","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">fifteen.experiments</span></code>","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">fifteen</span></code>","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">fifteen.utils._git</span></code>","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">fifteen.utils._hcb_print</span></code>","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">fifteen.utils._loop_metrics</span></code>","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">fifteen.utils._pdb_safety_net</span></code>","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">fifteen.utils._stopwatch</span></code>","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">fifteen.utils._timestamp</span></code>","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">fifteen.utils</span></code>","fifteen documentation"],titleterms:{"class":[1,2,4,6,7,8,9,13,17],"function":[0,3,5,6,11,12,13,14,15,16,17],_cycled_minibatch:0,_dataload:1,_experi:7,_git:11,_hcb_print:12,_in_memory_dataload:2,_log_data:8,_loop_metr:13,_pdb_safety_net:14,_prefetching_map:3,_protocol:4,_sharding_map:5,_stopwatch:15,_timestamp:16,api:18,attribut:[0,1,2,3,4,5,7,8,13],content:[0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17],data:[0,1,2,3,4,5,6],document:18,experi:[7,8,9],fifteen:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],modul:[0,1,2,3,4,5,7,8,11,12,13,14,15,16],packag:[6,9,17],refer:18,subpackag:10,util:[11,12,13,14,15,16,17]}})