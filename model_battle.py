#!/usr/bin/env python3
"""
五子棋AI模型对战系统（精简版）
两个模型进行一对一对战，支持GUI可视化
"""

import os
import glob
import time
import copy
import numpy as np
import asyncio
import platform
import pygame

import train as gomoku_cnn
from train import MCTS

class ModelBattle:
    def __init__(self):
        self.board_size = 15
        self.available_models = self.get_available_models()
        
    def get_available_models(self):
        """获取所有可用的模型文件"""
        models = []
        
        # 检查强化训练模型目录
        strong_path = 'gomoku_cnn_strong'
        if os.path.exists(strong_path):
            model_files = glob.glob(os.path.join(strong_path, '*.pth'))
            for model_file in model_files:
                filename = os.path.basename(model_file)
                if filename.startswith('backup_'):
                    continue  # 跳过备份文件
                try:
                    round_num = int(filename.split('.')[0])
                    models.append({
                        'path': model_file,
                        'name': f"第{round_num}轮模型",
                        'round': round_num,
                        'type': 'strong'
                    })
                except:
                    continue
        
        # 按轮次排序
        models.sort(key=lambda x: x['round'])
        return models
    
    def select_models(self):
        """选择对战的两个模型"""
        if len(self.available_models) < 2:
            print("❌ 至少需要2个模型才能进行对战！")
            return None, None
        
        print("🤖 可用模型列表：")
        print("="*60)
        for i, model in enumerate(self.available_models):
            print(f"{i+1:2d}. {model['name']:15s} - {model['path']}")
        print("="*60)
        
        # 选择第一个模型（黑棋）
        while True:
            try:
                choice1 = input("🔴 请选择第一个模型 (黑棋，先手): ").strip()
                if choice1 == "":
                    print("❌ 请输入有效选择")
                    continue
                choice1 = int(choice1)
                if 1 <= choice1 <= len(self.available_models):
                    model1 = self.available_models[choice1 - 1]
                    break
                else:
                    print(f"❌ 请输入 1-{len(self.available_models)} 之间的数字")
            except ValueError:
                print("❌ 请输入有效数字")
            except KeyboardInterrupt:
                print("\n👋 退出程序")
                return None, None
        
        # 选择第二个模型（白棋）
        while True:
            try:
                choice2 = input("⚪ 请选择第二个模型 (白棋，后手): ").strip()
                if choice2 == "":
                    print("❌ 请输入有效选择")
                    continue
                choice2 = int(choice2)
                if 1 <= choice2 <= len(self.available_models):
                    model2 = self.available_models[choice2 - 1]
                    break
                else:
                    print(f"❌ 请输入 1-{len(self.available_models)} 之间的数字")
            except ValueError:
                print("❌ 请输入有效数字")
            except KeyboardInterrupt:
                print("\n👋 退出程序")
                return None, None
        
        return model1, model2
    
    def play_battle_with_gui(self, model1, model2, simulations=200):
        """使用GUI进行模型对战"""
        print("\n" + "="*60)
        print("🎯 开始模型对战")
        print(f"🔴 黑棋: {model1['name']} ({model1['path']})")
        print(f"⚪ 白棋: {model2['name']} ({model2['path']})")
        print(f"🧠 MCTS模拟次数: {simulations}")
        print("="*60)
        
        try:
            # 创建专用的GUI对战类
            battle_gui = BattleGUI(
                model1_path=model1['path'], 
                model2_path=model2['path'],
                model1_name=model1['name'],
                model2_name=model2['name'],
                simulations=simulations
            )
            
            print("🎮 启动GUI对战界面...")
            
            # 运行GUI对战
            if platform.system() == "Emscripten":
                asyncio.ensure_future(battle_gui.main())
            else:
                asyncio.run(battle_gui.main())
                
        except Exception as e:
            print(f"❌ GUI对战出错: {e}")
            return None
    
    def start_battle(self):
        """开始一对一模型对战"""
        print("🎯 五子棋AI模型对战系统")
        print("只进行一场对战，支持GUI可视化")
        print()
        
        # 选择模型
        model1, model2 = self.select_models()
        if not model1 or not model2:
            return
        
        # 设置模拟次数
        while True:
            try:
                sim_input = input("🧠 设置MCTS模拟次数 (回车默认200): ").strip()
                if sim_input == "":
                    simulations = 200
                    break
                simulations = int(sim_input)
                if simulations > 0:
                    break
                else:
                    print("❌ 模拟次数必须大于0")
            except ValueError:
                print("❌ 请输入有效数字")
            except KeyboardInterrupt:
                print("\n👋 退出程序")
                return
        
        # 开始GUI对战
        self.play_battle_with_gui(model1, model2, simulations)


# 专用于模型对战的GUI类
class BattleGUI:
    def __init__(self, model1_path, model2_path, model1_name, model2_name, simulations=200):
        pygame.init()
        self.board_size = 15
        self.cell_size = 40
        self.margin = 20
        self.width = self.height = self.margin * 2 + self.cell_size * (self.board_size - 1)
        self.screen = pygame.display.set_mode((self.width, self.width + 150))
        self.clock = pygame.time.Clock()
        
        # 专门加载中文字体
        font_loaded = False
        
        # macOS 系统字体路径
        chinese_fonts = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/Library/Fonts/Arial Unicode.ttf"
        ]
        
        print("🔍 正在搜索中文字体...")
        for font_path in chinese_fonts:
            if os.path.exists(font_path):
                try:
                    self.font = pygame.font.Font(font_path, 20)
                    self.big_font = pygame.font.Font(font_path, 28)
                    print(f"✅ 成功加载中文字体: {os.path.basename(font_path)}")
                    font_loaded = True
                    break
                except Exception as e:
                    print(f"❌ 加载字体失败 {font_path}: {e}")
                    continue
        
        if not font_loaded:
            # 尝试使用 pygame 的系统字体
            print("🔍 尝试使用系统字体...")
            try:
                # 在 macOS 上尝试获取系统字体
                system_font = pygame.font.get_default_font()
                self.font = pygame.font.Font(system_font, 20)
                self.big_font = pygame.font.Font(system_font, 28)
                print(f"✅ 使用系统默认字体: {system_font}")
            except:
                self.font = pygame.font.Font(None, 24)
                self.big_font = pygame.font.Font(None, 32)
                print("⚠️ 使用 pygame 默认字体")

        # 游戏状态
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1  # 1为黑棋(model1)，-1为白棋(model2)
        self.move_history = []
        self.game_over = False
        self.winner = None
        self.move_count = 0
        
        # 模型信息
        self.model1_path = model1_path
        self.model2_path = model2_path
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.simulations = simulations
        
        # 初始化AI
        self.mcts1 = MCTS(model=model1_path, use_rand=0)  # 黑棋
        self.mcts2 = MCTS(model=model2_path, use_rand=0)  # 白棋
        
        # 显示信息
        print(f"🔴 黑棋模型加载完成: {model1_name}")
        print(f"📁 路径: {model1_path}")
        print(f"🔧 设备: {self.mcts1.device}")
        print()
        print(f"⚪ 白棋模型加载完成: {model2_name}")
        print(f"📁 路径: {model2_path}")
        print(f"🔧 设备: {self.mcts2.device}")
        print()
        
        # 颜色定义
        self.BACKGROUND = (220, 179, 92)
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 128, 0)
    
    def draw_board(self):
        """绘制棋盘"""
        self.screen.fill(self.BACKGROUND)
        
        # 绘制网格
        for i in range(self.board_size):
            # 横线
            start_pos = (self.margin, self.margin + i * self.cell_size)
            end_pos = (self.margin + (self.board_size - 1) * self.cell_size, self.margin + i * self.cell_size)
            pygame.draw.line(self.screen, self.BLACK, start_pos, end_pos, 1)
            
            # 竖线
            start_pos = (self.margin + i * self.cell_size, self.margin)
            end_pos = (self.margin + i * self.cell_size, self.margin + (self.board_size - 1) * self.cell_size)
            pygame.draw.line(self.screen, self.BLACK, start_pos, end_pos, 1)
        
        # 绘制棋子
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] != 0:
                    color = self.BLACK if self.board[i][j] == 1 else self.WHITE
                    center = (self.margin + j * self.cell_size, self.margin + i * self.cell_size)
                    pygame.draw.circle(self.screen, color, center, 15)
                    pygame.draw.circle(self.screen, self.BLACK, center, 15, 2)
    
    def draw_info(self):
        """绘制游戏信息"""
        y_start = self.height + 10
        
        # 当前玩家信息
        if not self.game_over:
            current_text = "当前回合: 黑棋" if self.current_player == 1 else "当前回合: 白棋"
            current_model = self.model1_name if self.current_player == 1 else self.model2_name
            try:
                text_surface = self.font.render(f"{current_text} ({current_model})", True, self.BLACK)
                self.screen.blit(text_surface, (10, y_start))
            except:
                # 如果中文渲染失败，使用英文
                text_surface = self.font.render(f"Current: {'Black' if self.current_player == 1 else 'White'} ({current_model})", True, self.BLACK)
                self.screen.blit(text_surface, (10, y_start))
        else:
            # 显示获胜者
            if self.winner == 1:
                try:
                    win_text = f"黑棋获胜! ({self.model1_name})"
                    text_surface = self.big_font.render(win_text, True, self.RED)
                except:
                    win_text = f"Black Wins! ({self.model1_name})"
                    text_surface = self.big_font.render(win_text, True, self.RED)
            elif self.winner == -1:
                try:
                    win_text = f"白棋获胜! ({self.model2_name})"
                    text_surface = self.big_font.render(win_text, True, self.BLUE)
                except:
                    win_text = f"White Wins! ({self.model2_name})"
                    text_surface = self.big_font.render(win_text, True, self.BLUE)
            else:
                try:
                    win_text = "平局!"
                    text_surface = self.big_font.render(win_text, True, self.GREEN)
                except:
                    win_text = "Draw!"
                    text_surface = self.big_font.render(win_text, True, self.GREEN)
            
            self.screen.blit(text_surface, (10, y_start))
        
        # 移动计数
        try:
            move_text = f"步数: {self.move_count}"
            text_surface = self.font.render(move_text, True, self.BLACK)
        except:
            move_text = f"Moves: {self.move_count}"
            text_surface = self.font.render(move_text, True, self.BLACK)
        self.screen.blit(text_surface, (10, y_start + 30))
        
        # 模型信息
        try:
            model_text = f"黑棋: {self.model1_name} vs 白棋: {self.model2_name}"
            text_surface = self.font.render(model_text, True, self.BLACK)
        except:
            model_text = f"Black: {self.model1_name} vs White: {self.model2_name}"
            text_surface = self.font.render(model_text, True, self.BLACK)
        self.screen.blit(text_surface, (10, y_start + 55))
        
        # 模拟次数
        try:
            sim_text = f"MCTS模拟次数: {self.simulations}"
            text_surface = self.font.render(sim_text, True, self.BLACK)
        except:
            sim_text = f"MCTS Simulations: {self.simulations}"
            text_surface = self.font.render(sim_text, True, self.BLACK)
        self.screen.blit(text_surface, (10, y_start + 80))
    
    def make_move(self, move):
        """执行移动"""
        if move is None or self.game_over:
            return False
        
        i, j = move
        if 0 <= i < self.board_size and 0 <= j < self.board_size and self.board[i][j] == 0:
            self.board[i][j] = self.current_player
            self.move_history.append((move, self.current_player))
            self.move_count += 1
            
            # 检查胜负
            if gomoku_cnn.evaluation_func(self.board):
                self.game_over = True
                self.winner = self.current_player
                print(f"游戏结束! {'黑棋' if self.current_player == 1 else '白棋'}获胜!")
                return True
            
            # 检查平局（棋盘满了）
            if np.count_nonzero(self.board) == self.board_size * self.board_size:
                self.game_over = True
                self.winner = 0
                print("游戏结束! 平局!")
                return True
            
            # 切换玩家
            self.current_player = -self.current_player
            return True
        
        return False
    
    def get_ai_move(self):
        """获取AI移动"""
        if self.game_over:
            return None
        
        # 根据当前玩家选择对应的MCTS
        mcts = self.mcts1 if self.current_player == 1 else self.mcts2
        player_name = self.model1_name if self.current_player == 1 else self.model2_name
        
        print(f"{player_name} 正在思考...")
        
        # 复制棋盘并根据需要翻转
        board_copy = copy.deepcopy(self.board)
        if self.current_player == -1:
            # 对于白棋，需要翻转棋盘视角
            board_copy = -board_copy
        
        # 运行MCTS
        start_time = time.time()
        result = mcts.run(board_copy, self.simulations, train=0, cur_root=None, return_root=1)
        _, root = result
        think_time = time.time() - start_time
        
        if root is None or not root.children:
            print(f"{player_name} 无法找到有效移动")
            return None
        
        # 获取访问次数最多的移动
        visit_counts = {}
        for move, (child, _) in root.children.items():
            if child is not None:
                visit_counts[move] = child.visit_count
        
        if not visit_counts:
            print(f"{player_name} 没有可用的移动")
            return None
        
        best_move = max(visit_counts.keys(), key=lambda m: visit_counts[m])
        best_visits = visit_counts[best_move]
        
        print(f"{player_name} 选择 {best_move}，访问次数: {best_visits}，用时: {think_time:.2f}秒")
        
        return best_move
    
    async def update_loop(self):
        """主更新循环"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and self.game_over:
                    # 空格键重新开始
                    self.__init__(self.model1_path, self.model2_path, 
                                self.model1_name, self.model2_name, self.simulations)
                    return True
        
        # AI自动移动
        if not self.game_over:
            move = self.get_ai_move()
            if move:
                self.make_move(move)
            else:
                # 无法移动，判定为平局
                self.game_over = True
                self.winner = 0
                print("无法继续移动，平局!")
        
        # 绘制界面
        self.draw_board()
        self.draw_info()
        
        pygame.display.flip()
        return True
    
    def setup(self):
        """设置界面"""
        pygame.display.set_caption("五子棋AI模型对战")
        
        # 测试中文字体渲染
        try:
            test_surface = self.font.render("测试中文", True, self.BLACK)
            print("✅ 中文字体渲染测试成功")
        except Exception as e:
            print(f"❌ 中文字体渲染测试失败: {e}")
    
    async def main(self):
        """主循环"""
        self.setup()
        running = True
        
        print("🎮 GUI对战开始!")
        print("游戏将自动进行，按ESC退出")
        if self.game_over:
            print("游戏结束后按空格键重新开始")
        
        while running:
            running = await self.update_loop()
            await asyncio.sleep(0.1)  # 稍微慢一点，便于观察
        
        pygame.quit()
        print("👋 GUI对战结束")


if __name__ == "__main__":
    battle = ModelBattle()
    battle.start_battle()
