To run:
1. change with_retract (and maybe with_reaching) to False in code
2. python main_sim_clean_test.py --seed xxxx --test_placing 0 --long_move 0 --use_height 0	(so it will run trials with 0411)
3. check the surrounding objs to see if they are arranged the way you want - my surrounding objs have a larger xy range then you do; they are white in my code
4. actually I think we should more densely sample from the first half (say ~35 steps) of each stacking. The first steps are far more important for success, and contains more tilted objs.
5. change mass and friction to random if you prefer

python main_sim_clean_test.py \
    --seed 1 \
    --test_placing 0 \
    --long_move 0 \
    --use_height 0
python main_sim_clean_test.py \
    --seed 1 \
    --test_placing 1 --long_move 0 --use_height 0
