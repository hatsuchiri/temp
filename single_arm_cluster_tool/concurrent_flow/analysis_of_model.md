1. recipe initialization
sdcfEnv.py
    class sdcfEnv
        reset()
            self._init_recipes()
                _create_recipe里取了random

2.有的参数设置会跑崩
会跑崩的参数举例：--foup_size 50 --prod_quantity 10 --done_quantity 50
                --foup_size 25 --prod_quantity 2 --done_quantity 25
                --foup_size 25 --prod_quantity 4 --done_quantity 100
不会跑崩的参数举例：--foup_size 50 --prod_quantity 10 --done_quantity 25
                --foup_size 25 --prod_quantity 1 --done_quantity 50 
                --foup_size 100 --prod_quantity 1 --done_quantity 50 