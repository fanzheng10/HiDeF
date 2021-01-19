net='BioPlex'
python ../hidef/hidef_finder.py --g $net\_sub1.tsv $net\_sub2.tsv --maxres 50 --o $net\_multiplex_test1 --iter --layer_weight 1 1 
python ../hidef/hidef_finder.py --g $net\_sub1.tsv $net\_sub2.tsv --maxres 50 --o $net\_multiplex_test2 --iter --layer_weight 1 -1

