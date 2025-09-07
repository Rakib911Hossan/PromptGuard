# run loop from 1 to 24 for num_turns, keep shots at 5
# for i in {1..24}; do
#     echo "Running for num_turns $i"
#     PYTHONPATH=. uv run src/few_shot.py --num_shots 5 --num_turns $i |& tee "logs/few_shot_num_turns_$i.log"
#     echo "Done for num_turns $i"
# done

# "Qwen/Qwen3-32B",
#     "Qwen/QwQ-32B",
#     "openai/gpt-oss-20b",
#     "openai/gpt-oss-120b",


# PYTHONPATH=. uv run src/few_shot.py --test --model_id "Qwen/Qwen3-30B-A3B-Thinking-2507" --num_turns 7 --num_shots 10 |& tee "logs/few_shot_submission_30B_A3B_Thinking_2507.log"
# PYTHONPATH=. uv run src/few_shot.py --test --model_id "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8" |& tee "logs/few_shot_submission_235B_A22B_Thinking_2507_FP8.log"
PYTHONPATH=. uv run src/few_shot.py --test --model_id "Qwen/Qwen3-32B" |& tee "logs/few_shot_submission_32B.log"
# PYTHONPATH=. uv run src/few_shot.py --test --model_id "Qwen/QwQ-32B" |& tee "logs/few_shot_submission_32B_QwQ.log"
PYTHONPATH=. uv run src/few_shot.py --test --model_id "openai/gpt-oss-20b" |& tee "logs/few_shot_submission_20b.log"
# PYTHONPATH=. uv run src/few_shot.py --test --model_id "openai/gpt-oss-120b" |& tee "logs/few_shot_submission_120b.log"