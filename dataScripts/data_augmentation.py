#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强脚本

用于扩充甄嬛传训练数据，提高模型性能
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from loguru import logger
import jieba
from collections import defaultdict

class DataAugmentation:
    """
    数据增强类，用于扩充训练数据
    """
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent
        
        # 同义词词典
        self.synonym_dict = {
            "皇上": ["陛下", "圣上", "万岁爷", "官家"],
            "臣妾": ["嫔妾", "妾身", "民女"],
            "娘娘": ["主子", "主位", "小主"],
            "本宫": ["本座", "哀家", "本妃"],
            "便是": ["就是", "即是", "正是"],
            "倒是": ["却是", "反是", "恰是"],
            "只是": ["不过", "然而", "可是"],
            "喜欢": ["喜爱", "钟爱", "青睐", "喜好"],
            "讨厌": ["厌恶", "憎恶", "厌烦"],
            "美丽": ["貌美", "美貌", "娇美", "秀美"],
            "聪慧": ["聪明", "机智", "伶俐", "机敏"],
            "温柔": ["温婉", "柔和", "柔顺", "温顺"],
            "高兴": ["欢喜", "愉悦", "欣喜", "喜悦"],
            "伤心": ["难过", "悲伤", "哀伤", "忧伤"],
            "生气": ["愤怒", "恼怒", "愤慨", "恼火"],
            "吃饭": ["用餐", "进食", "用膳"],
            "睡觉": ["安歇", "歇息", "就寝", "安寝"],
            "走路": ["行走", "踱步", "漫步", "缓步"],
            "说话": ["言语", "讲话", "言说", "道来"],
            "看": ["瞧", "望", "观", "瞅"],
            "听": ["闻", "聆听", "倾听"],
            "想": ["思虑", "思量", "思索", "寻思"],
            "知道": ["晓得", "明了", "清楚", "明白"],
            "不知道": ["不知", "不知晓", "不清楚", "不明白"],
            "好": ["甚好", "极好", "甚佳", "颇佳"],
            "不好": ["不佳", "不妥", "不甚好"],
            "可以": ["可行", "使得", "使得的"],
            "不可以": ["不可", "不行", "使不得"],
            "是": ["乃", "即", "为"],
            "不是": ["不是的", "并非", "非也"],
            "有": ["拥有", "持有", "具备"],
            "没有": ["无有", "不曾有", "未曾有"],
            "来了": ["到了", "抵达了", "到了这儿"],
            "走了": ["离去了", "离开了", "告退了"],
            "坐下": ["就座", "落座", "坐下来"],
            "起来": ["起身", "站起", "立起"],
            "请": ["恳请", "敬请", "乞请"],
            "谢谢": ["多谢", "感激", "谢过"],
            "对不起": ["抱歉", "对不住", "失礼了"],
            "没关系": ["无妨", "不妨事", "不打紧"],
            "为什么": ["为何", "何故", "因何"],
            "怎么样": ["如何", "怎样", "可好"],
            "多少": ["几何", "几许", "若干"],
            "什么时候": ["何时", "几时", "什么时候"],
            "哪里": ["何处", "何方", "哪儿"],
            "谁": ["何人", "谁人", "哪个"],
            "今天": ["今日", "今儿", "此日"],
            "明天": ["明日", "翌日", "来日"],
            "昨天": ["昨日", "前日", "昨儿"],
            "现在": ["此刻", "如今", "眼下"],
            "以后": ["日后", "此后", "往后"],
            "以前": ["从前", "以往", "昔日"],
            "这里": ["此处", "此地", "这里"],
            "那里": ["彼处", "彼地", "那里"],
            "这个": ["此物", "此件", "这个"],
            "那个": ["彼物", "彼件", "那个"],
            "什么": ["何物", "何事", "什么东西"],
            "怎么": ["如何", "怎样", "为何如此"],
            "非常": ["极为", "甚为", "格外", "分外"],
            "很": ["甚", "极", "颇为"],
            "有点": ["稍有", "略显", "些许"],
            "一点": ["些许", "少许", "一丝"],
            "全部": ["尽数", "全数", "统统"],
            "一些": ["些许", "一些些", "若干"],
            "很多": ["甚多", "诸多", "不少"],
            "很少": ["甚少", "鲜少", "寥寥"],
            "全部": ["尽数", "全数", "统统"],
            "部分": ["一些", "某些", "一部分"],
            "开始": ["起始", "开端", "伊始"],
            "结束": ["终结", "完毕", "终了"],
            "继续": ["继续下去", "延续", "接续"],
            "停止": ["停下", "终止", "中止"],
            "帮助": ["协助", "援助", "帮衬"],
            "困难": ["艰难", "困苦", "棘手"],
            "容易": ["简单", "轻易", "不难"],
            "困难": ["艰难", "困苦", "棘手"],
            "快乐": ["欢乐", "快活", "高兴"],
            "痛苦": ["痛苦不堪", "苦不堪言", "痛楚"],
            "忙碌": ["繁忙", "劳碌", "忙碌不堪"],
            "空闲": ["闲暇", "空暇", "清闲"],
            "重要": ["要紧", "紧要", "重大"],
            "不重要": ["不打紧", "无关紧要", "不要紧"],
            "正确": ["对的", "准确", "无误"],
            "错误": ["不对", "谬误", "有误"],
            "新": ["崭新", "全新", "新鲜"],
            "旧": ["陈旧", "老旧", "破旧"],
            "大": ["巨大", "庞大", "宏大"],
            "小": ["微小", "细小", "渺小"],
            "高": ["高大", "高耸", "挺拔"],
            "低": ["低矮", "矮小", "低下"],
            "长": ["修长", "细长", "长远"],
            "短": ["短小", "短暂", "短促"],
            "快": ["迅速", "急速", "飞快"],
            "慢": ["缓慢", "迟缓", "慢慢"],
            "热": ["炎热", "酷热", "燥热"],
            "冷": ["寒冷", "冰冷", "严寒"],
            "热": ["烫", "温热", "暖和"],
            "冷": ["凉", "冰凉", "清凉"],
            "好": ["优良", "优质", "上好"],
            "坏": ["恶劣", "糟糕", "不堪"],
            "干净": ["洁净", "清洁", "清爽"],
            "脏": ["污秽", "肮脏", "龌龊"],
            "美丽": ["漂亮", "好看", "俊美"],
            "丑陋": ["丑恶", "丑怪", "难看"],
            "年轻": ["青春", "年少", "青葱"],
            "年老": ["年迈", "老迈", "苍老"],
            "富有": ["富裕", "富足", "殷实"],
            "贫穷": ["贫困", "贫苦", "穷困"],
            "健康": ["康健", "健壮", "安康"],
            "生病": ["患病", "染病", "抱病"],
            "安全": ["安稳", "平安", "安定"],
            "危险": ["凶险", "险恶", "危机"],
            "光明": ["明亮", "光亮", "明朗"],
            "黑暗": ["昏暗", "漆黑", "阴暗"],
            "安静": ["宁静", "寂静", "静谧"],
            "吵闹": ["喧闹", "嘈杂", "喧哗"],
            "高兴": ["愉快", "开心", "欢喜"],
            "悲伤": ["忧伤", "哀伤", "愁苦"],
            "爱": ["钟爱", "喜爱", "疼爱"],
            "恨": ["憎恨", "仇恨", "怨恨"],
            "朋友": ["友人", "挚友", "知己"],
            "敌人": ["仇敌", "宿敌", "对手"],
            "家人": ["亲人", "家属", "骨肉"],
            "老师": ["师父", "师长", "夫子"],
            "学生": ["弟子", "门生", "徒儿"],
            "工作": ["差事", "营生", "勾当"],
            "休息": ["歇息", "休憩", "小憩"],
            "学习": ["修习", "研习", "温习"],
            "玩耍": ["嬉戏", "游玩", "游戏"],
            "吃饭": ["用餐", "就餐", "进食"],
            "喝水": ["饮水", "啜水", "品茗"],
            "睡觉": ["安睡", "就寝", "歇息"],
            "起床": ["起身", "醒转", "起来"],
            "穿衣": ["着装", "穿戴", "披挂"],
            "脱衣": ["宽衣", "卸装", "褪去"],
            "洗澡": ["沐浴", "浣洗", "濯足"],
            "走路": ["行路", "踱步", "迈步"],
            "跑步": ["奔走", "疾驰", "飞奔"],
            "坐下": ["落座", "就座", "坐定"],
            "站起": ["起立", "立起", "起身"],
            "开门": ["启门", "推门", "拉开"],
            "关门": ["闭门", "掩门", "合上"],
            "进来": ["进入", "步入", "踏进"],
            "出去": ["出去", "离去", "出门"],
            "买": ["购入", "采买", "置办"],
            "卖": ["出售", "售卖", "兜售"],
            "送": ["赠予", "奉上", "呈上"],
            "拿": ["取来", "拿来", "携来"],
            "放": ["放置", "摆放", "安置"],
            "看": ["观望", "凝视", "注视"],
            "听": ["聆听", "谛听", "倾听"],
            "说": ["言道", "说道", "言说"],
            "写": ["书写", "撰写", "挥毫"],
            "读": ["诵读", "研读", "品读"],
            "唱": ["吟唱", "讴歌", "咏唱"],
            "跳": ["跳跃", "腾跃", "蹦跳"],
            "笑": ["微笑", "浅笑", "莞尔"],
            "哭": ["啼哭", "哭泣", "泪下"],
            "生气": ["恼怒", "愤怒", "愤慨"],
            "害怕": ["恐惧", "畏惧", "惶恐"],
            "惊讶": ["惊异", "诧异", "愕然"],
            "高兴": ["欣喜", "喜悦", "欢快"],
            "失望": ["沮丧", "失落", "怅然"],
            "希望": ["期望", "盼望", "希冀"],
            "绝望": ["无望", "失望", "心灰意冷"],
            "相信": ["信服", "坚信", "笃信"],
            "怀疑": ["疑虑", "猜疑", "质疑"],
            "记住": ["铭记", "牢记", "谨记"],
            "忘记": ["遗忘", "忘却", "忘记"],
            "开始": ["着手", "开启", "发动"],
            "结束": ["完结", "收尾", "告终"],
            "继续": ["延续", "接续", "持续"],
            "停止": ["中止", "终止", "停下"],
            "改变": ["变更", "改动", "变换"],
            "保持": ["维持", "保留", "保持"],
            "失去": ["丢失", "丧失", "失去"],
            "得到": ["获得", "取得", "得到"],
            "寻找": ["寻觅", "搜寻", "查找"],
            "发现": ["发觉", "发现", "察觉"],
            "隐藏": ["隐匿", "藏匿", "遮掩"],
            "显示": ["显现", "展示", "显露"],
            "增加": ["增添", "增长", "增加"],
            "减少": ["减少", "削减", "缩减"],
            "上升": ["升高", "攀升", "上涨"],
            "下降": ["降低", "下滑", "下跌"],
            "前进": ["前行", "前进", "迈进"],
            "后退": ["后退", "退却", "撤退"],
            "上升": ["升起", "升腾", "上扬"],
            "下降": ["落下", "沉降", "下坠"],
            "打开": ["开启", "掀开", "拉开"],
            "关闭": ["闭合", "合上", "掩上"],
            "连接": ["相连", "联结", "衔接"],
            "分离": ["分开", "分离", "割裂"],
            "合并": ["合并", "融合", "合一"],
            "分裂": ["分裂", "分化", "瓦解"],
            "保护": ["护卫", "庇护", "保护"],
            "攻击": ["攻打", "进攻", "袭击"],
            "防御": ["防守", "抵御", "防备"],
            "逃跑": ["逃遁", "遁逃", "逃离"],
            "追赶": ["追逐", "追赶", "追击"],
            "帮助": ["援助", "协助", "帮衬"],
            "阻碍": ["阻挡", "妨碍", "阻挠"],
            "支持": ["支撑", "扶持", "支援"],
            "反对": ["反对", "抵制", "抗拒"],
            "同意": ["赞同", "应允", "允准"],
            "拒绝": ["拒绝", "推辞", "谢绝"],
            "接受": ["接纳", "收受", "接收"],
            "放弃": ["放弃", "舍弃", "抛弃"],
            "坚持": ["坚持", "执着", "坚守"],
            "放弃": ["丢弃", "舍弃", "抛弃"],
            "尝试": ["尝试", "试着", "试图"],
            "成功": ["成功", "得手", "达成"],
            "失败": ["失败", "败北", "落败"],
            "胜利": ["获胜", "取胜", "得胜"],
            "失败": ["败北", "失利", "落败"],
            "赢": ["获胜", "取胜", "赢得"],
            "输": ["输了", "败了", "输了"],
            "赢": ["胜出", "取胜", "获胜"],
            "输": ["败北", "失利", "落败"],
            "开始": ["启程", "出发", "动身"],
            "结束": ["完毕", "终结", "收场"],
            "继续": ["接着", "继而", "随后"],
            "停止": ["停下", "中止", "终止"],
            "等待": ["等候", "等待", "守候"],
            "离开": ["离去", "告辞", "辞别"],
            "到达": ["抵达", "来到", "至此"],
            "返回": ["归来", "回来", "返程"],
            "出发": ["启程", "动身", "出发"],
            "到达": ["抵达", "到达", "至此"],
            "旅行": ["游历", "游走", "出行"],
            "回家": ["归家", "回家", "返家"],
            "出门": ["外出", "出门", "出去"],
            "进来": ["进来", "进入", "踏入"],
            "出去": ["出去", "外出", "离开"],
            "上升": ["攀升", "上涨", "升高"],
            "下降": ["下滑", "下跌", "降低"],
            "增加": ["增长", "增多", "增添"],
            "减少": ["减少", "缩减", "削减"],
            "扩大": ["扩张", "扩展", "拓宽"],
            "缩小": ["收缩", "缩小", "缩减"],
            "加强": ["强化", "加强", "增强"],
            "减弱": ["削弱", "减弱", "减轻"],
            "改善": ["改进", "改良", "改善"],
            "恶化": ["恶化", "变坏", "恶化"],
            "解决": ["解决", "化解", "处置"],
            "产生": ["产生", "出现", "引发"],
            "消失": ["消失", "不见", "消散"],
            "出现": ["出现", "显现", "浮现"],
            "隐藏": ["隐藏", "隐匿", "藏匿"],
            "公开": ["公开", "公布", "披露"],
            "秘密": ["秘密", "隐秘", "机密"],
            "明显": ["明显", "显著", "突出"],
            "隐藏": ["隐蔽", "隐藏", "遮掩"],
            "重要": ["重要", "紧要", "重大"],
            "次要": ["次要", "枝节", "细务"],
            "主要": ["主要", "首要", "重要"],
            "次要": ["次要", "次之", "其次"],
            "必要": ["必要", "必需", "必须"],
            "不必要": ["不必要", "非必需", "不必"],
            "可能": ["可能", "也许", "或许"],
            "不可能": ["不可能", "断无可能", "绝无可能"],
            "肯定": ["肯定", "必定", "必然"],
            "否定": ["否定", "否认", "否决"],
            "真实": ["真实", "真切", "确实"],
            "虚假": ["虚假", "虚妄", "假的"],
            "正确": ["正确", "对的", "准确"],
            "错误": ["错误", "不对", "有误"],
            "美好": ["美好", "美妙", "优美"],
            "丑恶": ["丑恶", "丑陋", "恶毒"],
            "善良": ["善良", "仁善", "慈祥"],
            "邪恶": ["邪恶", "恶毒", "狠毒"],
            "诚实": ["诚实", "老实", "真诚"],
            "虚伪": ["虚伪", "虚假", "伪善"],
            "勇敢": ["勇敢", "英勇", "无畏"],
            "胆小": ["胆小", "怯懦", "畏缩"],
            "聪明": ["聪明", "聪慧", "机智"],
            "愚笨": ["愚笨", "愚昧", "蠢笨"],
            "勤奋": ["勤奋", "勤勉", "努力"],
            "懒惰": ["懒惰", "怠惰", "懒散"],
            "节约": ["节约", "节俭", "节省"],
            "浪费": ["浪费", "挥霍", "奢靡"],
            "整洁": ["整洁", "整齐", "干净"],
            "凌乱": ["凌乱", "杂乱", "混乱"],
            "有序": ["有序", "整齐", "井然"],
            "混乱": ["混乱", "紊乱", "杂乱"],
            "简单": ["简单", "简易", "容易"],
            "复杂": ["复杂", "繁复", "繁琐"],
            "容易": ["容易", "简单", "轻易"],
            "困难": ["困难", "艰难", "不易"],
            "便宜": ["便宜", "廉价", "实惠"],
            "昂贵": ["昂贵", "贵重", "奢侈"],
            "新鲜": ["新鲜", "新颖", "清新"],
            "陈旧": ["陈旧", "老旧", "过时"],
            "现代": ["现代", "当代", "现今"],
            "古代": ["古代", "古时", "往昔"],
            "现在": ["现在", "如今", "目前"],
            "过去": ["过去", "从前", "昔日"],
            "未来": ["未来", "将来", "日后"],
            "早上": ["清晨", "早晨", "晨曦"],
            "中午": ["中午", "正午", "晌午"],
            "晚上": ["晚上", "夜晚", "黄昏"],
            "春天": ["春季", "春日", "阳春"],
            "夏天": ["夏季", "夏日", "炎夏"],
            "秋天": ["秋季", "秋日", "金秋"],
            "冬天": ["冬季", "冬日", "严冬"],
            "晴天": ["晴日", "朗日", "丽日"],
            "雨天": ["雨日", "下雨天", "阴雨"],
            "雪天": ["雪日", "下雪天", "飘雪"],
            "风天": ["刮风天", "起风日", "风起"],
            "好天气": ["好天气", "晴好", "天朗气清"],
            "坏天气": ["坏天气", "恶劣天气", "天公不作美"]
        }
        
        # 古典语气词
        self.tone_words = ["呢", "啊", "呀", "吧", "嘛", "矣", "乎", "哉", "也"]
        
        # 甄嬛常用表达模式
        self.huanhuan_patterns = [
            "臣妾觉得{content}，不知皇上意下如何？",
            "皇上，{content}，臣妾斗胆进言。",
            "{content}，这便是臣妾的一点愚见。",
            "回皇上，{content}",
            "臣妾以为{content}，还请皇上明鉴。",
            "听闻{content}，臣妾心中甚慰。",
            "此事{content}，臣妾深以为然。"
        ]
        
    def synonym_replacement(self, text: str, ratio: float = 0.1) -> str:
        """
        同义词替换
        
        Args:
            text: 输入文本
            ratio: 替换比例
            
        Returns:
            替换后的文本
        """
        words = list(jieba.cut(text))
        num_to_replace = max(1, int(len(words) * ratio))
        
        # 找到可以替换的词
        replaceable_indices = []
        for i, word in enumerate(words):
            if word in self.synonym_dict:
                replaceable_indices.append(i)
        
        # 随机选择要替换的词
        if replaceable_indices:
            selected_indices = random.sample(replaceable_indices, 
                                            min(num_to_replace, len(replaceable_indices)))
            
            # 执行替换
            for idx in selected_indices:
                synonym_list = self.synonym_dict[words[idx]]
                words[idx] = random.choice(synonym_list)
        
        return ''.join(words)
    
    def tone_word_insertion(self, text: str, ratio: float = 0.1) -> str:
        """
        语气词插入
        
        Args:
            text: 输入文本
            ratio: 插入比例
            
        Returns:
            插入语气词后的文本
        """
        words = list(jieba.cut(text))
        num_to_insert = max(1, int(len(words) * ratio))
        
        # 随机选择插入位置
        for _ in range(num_to_insert):
            if words:  # 确保列表不为空
                insert_pos = random.randint(0, len(words) - 1)
                tone_word = random.choice(self.tone_words)
                words.insert(insert_pos, tone_word)
        
        return ''.join(words)
    
    def pattern_transformation(self, instruction: str, output: str) -> List[Dict]:
        """
        模式转换，生成更多表达方式
        
        Args:
            instruction: 原始指令
            output: 原始输出
            
        Returns:
            转换后的数据列表
        """
        transformed_data = []
        
        # 使用不同的表达模式生成新数据
        for pattern in self.huanhuan_patterns:
            new_output = pattern.format(content=output)
            transformed_data.append({
                "instruction": instruction,
                "input": "",
                "output": new_output
            })
        
        return transformed_data
    
    def augment_data(self, data: List[Dict], target_size: int = 2000) -> List[Dict]:
        """
        数据增强主函数
        
        Args:
            data: 原始数据
            target_size: 目标数据大小
            
        Returns:
            增强后的数据
        """
        augmented_data = data.copy()
        
        # 如果原始数据不足目标大小，则进行增强
        while len(augmented_data) < target_size:
            # 随机选择一条原始数据
            original_item = random.choice(data)
            instruction = original_item["instruction"]
            output = original_item["output"]
            
            # 应用不同的增强技术
            if random.random() < 0.5:  # 50%概率进行同义词替换
                new_instruction = self.synonym_replacement(instruction, 0.1)
                new_output = self.synonym_replacement(output, 0.1)
                augmented_data.append({
                    "instruction": new_instruction,
                    "input": "",
                    "output": new_output
                })
            
            if random.random() < 0.3:  # 30%概率插入语气词
                new_instruction = self.tone_word_insertion(instruction, 0.1)
                new_output = self.tone_word_insertion(output, 0.1)
                augmented_data.append({
                    "instruction": new_instruction,
                    "input": "",
                    "output": new_output
                })
            
            if random.random() < 0.2:  # 20%概率进行模式转换
                transformed_items = self.pattern_transformation(instruction, output)
                # 随机选择一个转换结果
                if transformed_items:
                    selected_item = random.choice(transformed_items)
                    augmented_data.append(selected_item)
        
        # 确保不超过目标大小
        if len(augmented_data) > target_size:
            augmented_data = random.sample(augmented_data, target_size)
        
        return augmented_data
    
    def generate_conversation_data(self, base_data: List[Dict], num_conversations: int = 500) -> List[Dict]:
        """
        生成多轮对话数据
        
        Args:
            base_data: 基础数据
            num_conversations: 要生成的对话数量
            
        Returns:
            多轮对话数据
        """
        conversation_data = []
        
        for _ in range(num_conversations):
            # 随机选择3-5轮对话
            conversation_length = random.randint(3, 5)
            selected_items = random.sample(base_data, conversation_length)
            
            # 构建对话历史
            history = []
            for i, item in enumerate(selected_items):
                if i == 0:
                    # 第一轮使用原始格式
                    conversation_data.append({
                        "instruction": item["instruction"],
                        "input": item["input"],
                        "output": item["output"]
                    })
                else:
                    # 后续轮次包含历史对话
                    history_prompt = "之前的对话：\n"
                    for j, hist_item in enumerate(selected_items[:i]):
                        history_prompt += f"皇上：{hist_item['instruction']}\n"
                        history_prompt += f"甄嬛：{hist_item['output']}\n"
                    
                    history_prompt += f"\n当前问题：{item['instruction']}"
                    
                    conversation_data.append({
                        "instruction": history_prompt,
                        "input": "",
                        "output": item["output"]
                    })
        
        return conversation_data

def main():
    """
    主函数
    """
    logger.info("开始数据增强...")
    
    # 初始化数据增强器
    augmenter = DataAugmentation()
    
    # 加载原始数据
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    raw_data_path = project_root / "data" / "raw" / "huanhuan.json"
    
    try:
        with open(raw_data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        logger.info(f"加载原始数据: {len(raw_data)} 条")
    except Exception as e:
        logger.error(f"加载原始数据失败: {e}")
        return
    
    # 验证数据格式
    valid_data = []
    for item in raw_data:
        if isinstance(item, dict) and all(key in item for key in ['instruction', 'input', 'output']):
            valid_data.append({
                "instruction": item['instruction'],
                "input": item['input'],
                "output": item['output']
            })
    
    logger.info(f"有效数据: {len(valid_data)} 条")
    
    # 数据增强到2000条
    logger.info("正在进行数据增强...")
    augmented_data = augmenter.augment_data(valid_data, target_size=2000)
    logger.info(f"增强后数据: {len(augmented_data)} 条")
    
    # 生成多轮对话数据
    logger.info("正在生成多轮对话数据...")
    conversation_data = augmenter.generate_conversation_data(valid_data, num_conversations=300)
    logger.info(f"生成多轮对话数据: {len(conversation_data)} 条")
    
    # 合并所有数据
    final_data = augmented_data + conversation_data
    logger.info(f"最终数据总量: {len(final_data)} 条")
    
    # 保存增强后的数据
    output_path = project_root / "data" / "processed" / "augmented_train.jsonl"
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in final_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"增强数据已保存至: {output_path}")
    except Exception as e:
        logger.error(f"保存增强数据失败: {e}")
        return
    
    # 创建训练/验证/测试分割
    random.shuffle(final_data)
    total_size = len(final_data)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)
    
    train_data = final_data[:train_size]
    val_data = final_data[train_size:train_size + val_size]
    test_data = final_data[train_size + val_size:]
    
    # 保存分割后的数据
    splits = [
        ("train.jsonl", train_data),
        ("validation.jsonl", val_data),
        ("test.jsonl", test_data)
    ]
    
    for filename, data_split in splits:
        split_path = project_root / "data" / "processed" / filename
        try:
            with open(split_path, 'w', encoding='utf-8') as f:
                for item in data_split:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info(f"{filename} 已保存: {len(data_split)} 条数据")
        except Exception as e:
            logger.error(f"保存 {filename} 失败: {e}")

if __name__ == "__main__":
    main()