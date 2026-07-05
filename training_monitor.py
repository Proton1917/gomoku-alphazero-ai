#!/usr/bin/env python3
"""
训练监控脚本 - 监控强化训练进度并测试模型棋力
"""
import os
import time
import torch
from ai_battle import play_game, Model
import glob

def check_training_progress():
    """检查训练进度"""
    strong_model_dir = 'gomoku_cnn_strong'
    if not os.path.exists(strong_model_dir):
        print("❌ 强化训练目录不存在")
        return 0
    
    model_files = glob.glob(os.path.join(strong_model_dir, '[0-9]*.pth'))
    model_numbers = []
    for f in model_files:
        try:
            num = int(os.path.basename(f).split('.')[0])
            model_numbers.append(num)
        except:
            continue
    
    if not model_numbers:
        print("📊 训练尚未完成第一轮")
        return 0
    
    latest_round = max(model_numbers)
    print(f"📈 训练进度: {latest_round}/50 轮完成 ({latest_round/50*100:.1f}%)")
    return latest_round

def test_model_strength(model_path, games=10):
    """测试模型棋力对比原始强模型"""
    print(f"🧠 测试模型: {model_path}")
    
    # 测试新模型 vs model_4090_trained.pth
    try:
        # 加载新模型
        new_model = Model(model_path)
        
        # 加载原始强模型
        strong_model = Model('model_4090_trained.pth')
        
        wins = 0
        for i in range(games):
            # 新模型执黑，强模型执白
            result = play_game(new_model, strong_model)
            if result == 1:  # 新模型获胜
                wins += 1
            print(f"  第{i+1}局: {'🟫新模型胜' if result == 1 else ('⚪强模型胜' if result == -1 else '🟡平局')}")
        
        win_rate = wins / games * 100
        print(f"🏆 新模型胜率: {win_rate:.1f}% ({wins}/{games})")
        
        if win_rate >= 40:
            print("🎉 棋力显著提升！")
        elif win_rate >= 20:
            print("📈 棋力有所改善")
        else:
            print("📉 仍需继续训练")
            
        return win_rate
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return 0

def main():
    """主监控循环"""
    print("🔍 五子棋强化训练监控器启动")
    print("=" * 50)
    
    last_checked_round = 0
    
    while True:
        current_round = check_training_progress()
        
        # 如果有新的训练轮次完成，进行测试
        if current_round > last_checked_round and current_round > 0:
            # 测试每10轮或最新轮次
            if current_round % 10 == 0 or current_round == 1:
                model_path = f'gomoku_cnn_strong/{current_round}.pth'
                if os.path.exists(model_path):
                    print(f"\n🧪 测试第{current_round}轮模型...")
                    win_rate = test_model_strength(model_path, games=5)
                    print(f"第{current_round}轮模型胜率: {win_rate:.1f}%\n")
            
            last_checked_round = current_round
        
        # 检查是否训练完成
        if current_round >= 50:
            print("🎉 训练完成！开始最终测试...")
            final_model = f'gomoku_cnn_strong/50.pth'
            if os.path.exists(final_model):
                print("🏆 最终模型棋力测试 (20局)")
                test_model_strength(final_model, games=20)
            break
        
        # 每5分钟检查一次
        time.sleep(300)

if __name__ == "__main__":
    main()
