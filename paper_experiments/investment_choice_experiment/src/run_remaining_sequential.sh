#!/bin/bash
# 남은 실험 순차 실행 스크립트
# 현재 실험 완료 후 실행하세요

cd /home/ubuntu/llm_addiction/investment_choice_extended_cot/src

echo "=========================================="
echo "남은 실험 순차 실행 시작"
echo "=========================================="

# $30 variable (4개)
echo "[1/22] $30 variable - GPT-4o"
python run_all_experiments.py --model gpt4o --constraint 30 --bet_type variable

echo "[2/22] $30 variable - GPT-4.1"
python run_all_experiments.py --model gpt41 --constraint 30 --bet_type variable

echo "[3/22] $30 variable - Claude"
python run_all_experiments.py --model claude --constraint 30 --bet_type variable

echo "[4/22] $30 variable - Gemini"
python run_all_experiments.py --model gemini --constraint 30 --bet_type variable

# $50 fixed (4개)
echo "[5/22] $50 fixed - GPT-4o"
python run_all_experiments.py --model gpt4o --constraint 50 --bet_type fixed

echo "[6/22] $50 fixed - GPT-4.1"
python run_all_experiments.py --model gpt41 --constraint 50 --bet_type fixed

echo "[7/22] $50 fixed - Claude"
python run_all_experiments.py --model claude --constraint 50 --bet_type fixed

echo "[8/22] $50 fixed - Gemini"
python run_all_experiments.py --model gemini --constraint 50 --bet_type fixed

# $50 variable (4개)
echo "[9/22] $50 variable - GPT-4o"
python run_all_experiments.py --model gpt4o --constraint 50 --bet_type variable

echo "[10/22] $50 variable - GPT-4.1"
python run_all_experiments.py --model gpt41 --constraint 50 --bet_type variable

echo "[11/22] $50 variable - Claude"
python run_all_experiments.py --model claude --constraint 50 --bet_type variable

echo "[12/22] $50 variable - Gemini"
python run_all_experiments.py --model gemini --constraint 50 --bet_type variable

# $70 fixed (4개)
echo "[13/22] $70 fixed - GPT-4o"
python run_all_experiments.py --model gpt4o --constraint 70 --bet_type fixed

echo "[14/22] $70 fixed - GPT-4.1"
python run_all_experiments.py --model gpt41 --constraint 70 --bet_type fixed

echo "[15/22] $70 fixed - Claude"
python run_all_experiments.py --model claude --constraint 70 --bet_type fixed

echo "[16/22] $70 fixed - Gemini"
python run_all_experiments.py --model gemini --constraint 70 --bet_type fixed

# $70 variable (4개)
echo "[17/22] $70 variable - GPT-4o"
python run_all_experiments.py --model gpt4o --constraint 70 --bet_type variable

echo "[18/22] $70 variable - GPT-4.1"
python run_all_experiments.py --model gpt41 --constraint 70 --bet_type variable

echo "[19/22] $70 variable - Claude"
python run_all_experiments.py --model claude --constraint 70 --bet_type variable

echo "[20/22] $70 variable - Gemini"
python run_all_experiments.py --model gemini --constraint 70 --bet_type variable

echo "=========================================="
echo "모든 실험 완료!"
echo "=========================================="
