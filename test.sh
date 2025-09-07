# run loop from 1 to 24 for num_turns, keep shots at 5
for i in {1..24}; do
    echo "Running for num_turns $i"
    PYTHONPATH=. uv run src/few_shot.py --num_shots 5 --num_turns $i |& tee "logs/few_shot_num_turns_$i.log"
    echo "Done for num_turns $i"
done