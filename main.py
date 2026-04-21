#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
声纹识别工作流程 - 基于ModelScope的声纹特征提取与相似度计算

本程序实现了从音频文件中提取声纹特征，并计算不同音频间的声纹相似度。
支持多种音频格式输入，包括.wav、.mp3、.flac、.m4a、.ogg、.aac等。
使用ModelScope框架加载iic/speech_eres2net_sv_zh-cn_16k-common声纹识别模型，
专门针对中文语音的16kHz常见声纹识别任务进行了优化。

功能包括：
- 音频文件加载与预处理（重采样至16kHz、单声道转换）
- 音频特征提取（梅尔频谱、RMS能量、过零率）
- 声纹特征向量提取
- 声纹相似度计算与排序

使用方法：
将待处理的音频文件放入input文件夹，将目标说话人样本放入target文件夹，运行main.py即可得到结果。

作者：pfzlsad
日期：2026.04.21
版本：1.0

版权所有 (C) 2026 pfzlsad

本程序是自由软件：你可以再分发它以及/或按照
GNU宽通用公共许可证（GNU General Public License v3.0）的要求进行
修改，正如许可证所写。许可证的副本请见文件 COPYING 或 <https://www.gnu.org/licenses/>。

本程序分发的目的是希望它有用，但没有任何担保；甚至没有
适销性或特定用途适用性的暗示担保。详见GNU宽通用公共许可证
获取更多信息。
"""


"""
声纹识别完整工作流程
功能：音频预处理 -> 精细切分 -> 多目标声纹比对 -> 结果整理
输入为：input文件夹中的音频文件，和target文件夹中的一个或多个目标说话人样本
输出为：target_speaker文件夹中的目标说话人片段，以及analysis_reports文件夹中的分析报告
"""

import os
import numpy as np
import librosa
import soundfile as sf
import warnings
import json
import shutil
import tempfile
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from modelscope.pipelines import pipeline

# 忽略警告
warnings.filterwarnings('ignore')

# ==================== 1. 音频预处理模块 ====================
def check_and_convert_audio(input_path, output_dir="converted_16k", target_sr=16000):
    """
    检测音频采样率，如果不是目标采样率则进行转换
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 先读取文件头信息检测采样率
        with sf.SoundFile(input_path) as audio_file:
            original_sr = audio_file.samplerate
        
        # 如果已经是目标采样率，直接返回原文件
        if original_sr == target_sr:
            print(f"✅ 无需转换: {os.path.basename(input_path)} (已是{target_sr}Hz)")
            return input_path
        
        # 需要转换的情况
        file_name = Path(input_path).stem + f"_16k.wav"
        output_path = os.path.join(output_dir, file_name)
        
        # 加载并重采样
        y, _ = librosa.load(input_path, sr=target_sr, mono=True)
        
        # 保存为16kHz WAV
        sf.write(output_path, y, target_sr)
        print(f"✅ 转换成功: {os.path.basename(input_path)} ({original_sr}Hz -> {target_sr}Hz)")
        return output_path
        
    except Exception as e:
        print(f"❌ 处理失败 {input_path}: {e}")
        return None

def convert_audio_to_16k(input_path, output_dir="converted_16k", sr=16000):
    """
    将音频转换为16kHz单声道WAV格式
    """
    return check_and_convert_audio(input_path, output_dir, sr)

def prepare_target_audios(target_audios, output_dir="converted_16k"):
    """
    准备目标音频，确保都是16kHz
    """
    prepared_targets = []
    
    for target_audio in target_audios:
        if not os.path.exists(target_audio):
            print(f"❌ 目标音频不存在: {target_audio}")
            continue
            
        converted_target = check_and_convert_audio(target_audio, output_dir)
        if converted_target:
            prepared_targets.append(converted_target)
    
    return prepared_targets

def batch_convert_audio(input_dir, output_dir="converted_16k"):
    """
    批量转换音频文件
    """
    converted_files = []
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
    
    print("=" * 60)
    print("音频预处理：采样率检测与16kHz转换")
    print("=" * 60)
    
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return converted_files
    
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in audio_extensions):
            converted = check_and_convert_audio(file_path, output_dir)
            if converted:
                converted_files.append(converted)
    
    print(f"\n✅ 预处理完成！共处理 {len(converted_files)} 个文件")
    return converted_files

# ==================== 2. 精细切分模块 ====================
def detect_acoustic_changes(energy_norm, zcr_norm, spectral_centroid, 
                           energy_thresh=0.3, zcr_thresh=0.4):
    """
    检测声学特征变化点
    """
    changes = []
    
    # 计算特征差分
    energy_diff = np.abs(np.diff(energy_norm))
    zcr_diff = np.abs(np.diff(zcr_norm))
    spec_diff = np.abs(np.diff(spectral_centroid / np.max(spectral_centroid)))
    
    # 检测显著变化点
    for i in range(1, len(energy_diff)-1):
        energy_change = energy_diff[i] > energy_thresh
        zcr_change = zcr_diff[i] > zcr_thresh
        spec_change = spec_diff[i] > 0.2
        
        # 如果多个特征同时变化，可能是说话人切换
        if (energy_change and zcr_change) or (energy_change and spec_change):
            changes.append(i)
    
    return changes

def fine_split_by_speaker(input_path, output_dir, sr=16000, 
                         min_segment_duration=1.5,
                         max_segment_duration=8.0,
                         min_silence_len=300,
                         silence_thresh_db=-40,
                         energy_variation_thresh=0.3,
                         zcr_variation_thresh=0.4):
    """
    精细切分音频，尽量分离不同说话人
    """
    print(f"\n精细切分: {os.path.basename(input_path)}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载音频
    y, sr = librosa.load(input_path, sr=sr, mono=True)
    duration = len(y) / sr
    
    # 计算音频特征
    hop_length = 256
    frame_length = 1024
    
    # 1. 多种特征计算
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    
    # 2. 归一化特征
    db_norm = (db - np.min(db)) / (np.max(db) - np.min(db) + 1e-10)
    zcr_norm = (zcr - np.min(zcr)) / (np.max(zcr) - np.min(zcr) + 1e-10)
    
    # 3. 静音检测
    silence_threshold = np.percentile(db, 30)
    is_silent = db < silence_threshold
    
    # 4. 检测声学特征变化点
    change_points = detect_acoustic_changes(db_norm, zcr_norm, spectral_centroid, 
                                          energy_variation_thresh, zcr_variation_thresh)
    
    # 5. 基于静音和变化点切分
    all_cut_points = set()
    
    # 添加静音切分点
    for i in range(1, len(is_silent)):
        if is_silent[i] and not is_silent[i-1]:
            silence_duration = 0
            j = i
            while j < len(is_silent) and is_silent[j]:
                silence_duration += 1
                j += 1
            
            if silence_duration * hop_length / sr >= min_silence_len / 1000:
                all_cut_points.add(i)
    
    # 添加声学变化点
    for point in change_points:
        all_cut_points.add(point)
    
    # 6. 排序并处理切分点
    cut_points = sorted(list(all_cut_points))
    
    # 7. 生成切分片段
    segments = []
    segment_counter = 0
    last_cut = 0
    
    for i, cut in enumerate(cut_points):
        start_frame = last_cut
        end_frame = cut
        segment_duration = (end_frame - start_frame) * hop_length / sr
        
        if segment_duration >= min_segment_duration:
            start_sample = start_frame * hop_length
            end_sample = end_frame * hop_length
            
            # 如果片段太长，强制切分
            if segment_duration > max_segment_duration:
                num_splits = int(np.ceil(segment_duration / max_segment_duration))
                split_duration = segment_duration / num_splits
                
                for j in range(num_splits):
                    sub_start = start_sample + int(j * split_duration * sr)
                    sub_end = min(end_sample, start_sample + int((j + 1) * split_duration * sr))
                    sub_duration = (sub_end - sub_start) / sr
                    
                    if sub_duration >= min_segment_duration:
                        segment_counter += 1
                        # 计算实际时间
                        actual_start_time = sub_start / sr
                        actual_end_time = sub_end / sr
                        
                        # 生成带时间标注的文件名
                        output_path = os.path.join(output_dir, 
                                                  f"segment_{segment_counter:04d}_{actual_start_time:.1f}s-{actual_end_time:.1f}s.wav")
                        sf.write(output_path, y[sub_start:sub_end], sr)
                        
                        segments.append({
                            'index': segment_counter,
                            'start_time': actual_start_time,
                            'end_time': actual_end_time,
                            'duration': sub_duration,
                            'file_path': output_path
                        })
            else:
                segment_counter += 1
                # 计算实际时间
                actual_start_time = start_sample / sr
                actual_end_time = end_sample / sr
                
                # 生成带时间标注的文件名
                output_path = os.path.join(output_dir, 
                                          f"segment_{segment_counter:04d}_{actual_start_time:.1f}s-{actual_end_time:.1f}s.wav")
                sf.write(output_path, y[start_sample:end_sample], sr)
                
                segments.append({
                    'index': segment_counter,
                    'start_time': actual_start_time,
                    'end_time': actual_end_time,
                    'duration': segment_duration,
                    'file_path': output_path
                })
        
        last_cut = cut
    
    # 处理最后一段
    if last_cut < len(db):
        start_frame = last_cut
        end_frame = len(db)
        segment_duration = (end_frame - start_frame) * hop_length / sr
        
        if segment_duration >= min_segment_duration:
            start_sample = start_frame * hop_length
            end_sample = end_frame * hop_length
            
            if segment_duration > max_segment_duration:
                num_splits = int(np.ceil(segment_duration / max_segment_duration))
                split_duration = segment_duration / num_splits
                
                for j in range(num_splits):
                    sub_start = start_sample + int(j * split_duration * sr)
                    sub_end = min(end_sample, start_sample + int((j + 1) * split_duration * sr))
                    sub_duration = (sub_end - sub_start) / sr
                    
                    if sub_duration >= min_segment_duration:
                        segment_counter += 1
                        # 计算实际时间
                        actual_start_time = sub_start / sr
                        actual_end_time = sub_end / sr
                        
                        # 生成带时间标注的文件名
                        output_path = os.path.join(output_dir, 
                                                  f"segment_{segment_counter:04d}_{actual_start_time:.1f}s-{actual_end_time:.1f}s.wav")
                        sf.write(output_path, y[sub_start:sub_end], sr)
                        
                        segments.append({
                            'index': segment_counter,
                            'start_time': actual_start_time,
                            'end_time': actual_end_time,
                            'duration': sub_duration,
                            'file_path': output_path
                        })
            else:
                segment_counter += 1
                # 计算实际时间
                actual_start_time = start_sample / sr
                actual_end_time = end_sample / sr
                
                # 生成带时间标注的文件名
                output_path = os.path.join(output_dir, 
                                          f"segment_{segment_counter:04d}_{actual_start_time:.1f}s-{actual_end_time:.1f}s.wav")
                sf.write(output_path, y[start_sample:end_sample], sr)
                
                segments.append({
                    'index': segment_counter,
                    'start_time': actual_start_time,
                    'end_time': actual_end_time,
                    'duration': segment_duration,
                    'file_path': output_path
                })
    
    print(f"  → 生成 {len(segments)} 个片段")
    return segments

def batch_fine_split(converted_files, output_base_dir="split_audio"):
    """
    批量精细切分
    """
    all_segments = []
    
    print("\n" + "=" * 60)
    print("音频精细切分")
    print("=" * 60)
    
    for audio_file in converted_files:
        file_name = Path(audio_file).stem
        output_dir = os.path.join(output_base_dir, file_name)
        
        segments = fine_split_by_speaker(
            input_path=audio_file,
            output_dir=output_dir,
            sr=16000,
            min_segment_duration=1.5,
            max_segment_duration=8.0
        )
        
        for seg in segments:
            seg['source_file'] = file_name
        
        all_segments.extend(segments)
    
    print(f"\n✅ 切分完成！共得到 {len(all_segments)} 个片段")
    return all_segments

# ==================== 3. 多目标声纹比对模块 ====================
def verify_with_multiple_targets(segment_audio, target_audios, sv_pipeline, threshold=0.6, strategy='mean'):
    """
    使用多个target进行声纹比对
    """
    similarities = []
    is_same_list = []
    
    for target_audio in target_audios:
        try:
            result = sv_pipeline([target_audio, segment_audio])
            similarity = result.get('score', 0.0)
            is_same_person = result.get('text', '').lower() == 'yes'
            
            similarities.append(similarity)
            is_same_list.append(is_same_person)
        except Exception as e:
            print(f"⚠️  target比对失败 {os.path.basename(target_audio)}: {e}")
            similarities.append(0.0)
            is_same_list.append(False)
    
    # 不同策略
    if strategy == 'mean':
        final_similarity = np.mean(similarities) if similarities else 0.0
        final_is_same = (sum(is_same_list) / len(is_same_list) > 0.5) if is_same_list else False
    elif strategy == 'max':
        final_similarity = np.max(similarities) if similarities else 0.0
        final_is_same = any(is_same_list) if is_same_list else False
    elif strategy == 'vote':
        final_similarity = np.mean(similarities) if similarities else 0.0
        final_is_same = (sum(is_same_list) > len(is_same_list) / 2) if is_same_list else False
    
    is_target_speaker = final_similarity > threshold and final_is_same
    
    return {
        'similarity': float(final_similarity),  # 转换为Python float
        'similarities': [float(s) for s in similarities],  # 转换为Python float列表
        'is_same_person': bool(final_is_same),  # 转换为Python bool
        'is_target_speaker': bool(is_target_speaker),  # 转换为Python bool
        'strategy': strategy
    }

def verify_all_segments_multi(all_segments, target_audios, sv_pipeline, 
                             threshold=0.6, strategy='mean', show_all=False):
    """
    批量多target声纹比对
    """
    verification_results = []
    target_count = 0
    
    print("\n" + "=" * 60)
    print("多目标声纹比对")
    print(f"目标样本数: {len(target_audios)}")
    print(f"比对策略: {strategy}")
    print(f"阈值: {threshold}")
    print("=" * 60)
    
    for i, segment in enumerate(all_segments):
        segment_path = segment['file_path']
        
        # 显示进度
        if show_all or (i + 1) % 10 == 0 or i == 0 or i == len(all_segments) - 1:
            print(f"比对 {i+1}/{len(all_segments)}: {os.path.basename(segment_path)}", end=" ")
        
        # 多target比对
        result = verify_with_multiple_targets(segment_path, target_audios, sv_pipeline, threshold, strategy)
        
        # 添加比对结果
        # 确保所有数值类型都是Python原生类型
        segment.update({
            'similarity': float(result['similarity']),
            'similarities': [float(s) for s in result['similarities']],
            'is_same_person': bool(result['is_same_person']),
            'is_target_speaker': bool(result['is_target_speaker']),
            'strategy': result['strategy']
        })
        
        verification_results.append(segment)
        
        if result['is_target_speaker']:
            target_count += 1
            if show_all or (i + 1) % 10 == 0 or i == 0 or i == len(all_segments) - 1:
                print(f"✅ 目标 (相似度: {result['similarity']:.3f})")
        else:
            if show_all or (i + 1) % 10 == 0 or i == 0 or i == len(all_segments) - 1:
                print(f"❌ 非目标 (相似度: {result['similarity']:.3f})")
    
    print(f"\n✅ 比对完成！目标说话人片段: {target_count}/{len(all_segments)}")
    return verification_results

# ==================== 4. 结果整理与保存模块 ====================
def save_target_speaker_segments(verification_results, output_dir="target_speaker"):
    """
    保存目标说话人片段，文件名包含原始时间线
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 筛选目标说话人片段
    target_segments = [seg for seg in verification_results if seg.get('is_target_speaker', False)]
    
    print(f"\n保存目标说话人片段 ({len(target_segments)} 个)")
    
    for i, segment in enumerate(target_segments):
        original_path = segment['file_path']
        
        # 生成新的文件名，包含原始时间线
        similarity = float(segment.get('similarity', 0.0))
        start_time = float(segment.get('start_time', 0.0))
        end_time = float(segment.get('end_time', 0.0))
        duration = float(segment.get('duration', 0.0))
        source = segment.get('source_file', 'unknown')
        
        # 移除原文件名中的时间信息，避免重复
        base_source = source.replace("_16k", "")
        
        # 新文件名格式：target_序号_原文件名_开始时间-结束时间_时长_相似度.wav
        new_name = f"target_{i+1:04d}_{base_source}_{start_time:.1f}s-{end_time:.1f}s_dur{duration:.1f}s_sim{similarity:.3f}.wav"
        new_path = os.path.join(output_dir, new_name)
        
        try:
            shutil.copy2(original_path, new_path)
            if (i + 1) % 20 == 0 or i == 0 or i == len(target_segments) - 1:
                print(f"✅ 保存 {i+1}/{len(target_segments)}: {new_name}")
        except Exception as e:
            print(f"❌ 保存失败 {i+1}: {e}")
    
    print(f"\n🎯 已保存 {len(target_segments)} 个目标说话人片段到: {output_dir}")
    return target_segments

def generate_comprehensive_report(verification_results, target_segments, target_audios, 
                                output_dir="reports", strategy='mean', threshold=0.6):
    """
    生成综合分析报告
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 文本报告
    report_path = os.path.join(output_dir, "analysis_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("声纹识别综合分析报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("配置参数:\n")
        f.write("-" * 40 + "\n")
        f.write(f"比对策略: {strategy}\n")
        f.write(f"相似度阈值: {threshold}\n")
        f.write(f"目标样本数: {len(target_audios)}\n")
        for i, target in enumerate(target_audios):
            f.write(f"  目标{i+1}: {os.path.basename(target)}\n")
        f.write("\n")
        
        f.write("总体统计:\n")
        f.write("-" * 40 + "\n")
        f.write(f"总片段数: {len(verification_results)}\n")
        f.write(f"目标说话人片段数: {len(target_segments)}\n")
        f.write(f"目标说话人占比: {len(target_segments)/len(verification_results)*100:.1f}%\n\n")
        
        # 时长分析
        durations = [float(seg.get('duration', 0)) for seg in verification_results]
        f.write("片段时长分析:\n")
        f.write("-" * 40 + "\n")
        f.write(f"平均时长: {np.mean(durations):.2f}秒\n")
        f.write(f"最短时长: {np.min(durations):.2f}秒\n")
        f.write(f"最长时长: {np.max(durations):.2f}秒\n")
        f.write(f"总音频时长: {np.sum(durations):.2f}秒\n\n")
        
        # 相似度分析
        similarities = [float(seg.get('similarity', 0)) for seg in verification_results]
        f.write("相似度分析:\n")
        f.write("-" * 40 + "\n")
        f.write(f"平均相似度: {np.mean(similarities):.3f}\n")
        f.write(f"最高相似度: {np.max(similarities):.3f}\n")
        f.write(f"最低相似度: {np.min(similarities):.3f}\n")
        f.write(f"相似度标准差: {np.std(similarities):.3f}\n\n")
        
        # 多target一致性分析
        if 'similarities' in verification_results[0]:
            f.write("多目标一致性分析:\n")
            f.write("-" * 40 + "\n")
            
            for i in range(len(target_audios)):
                target_sims = []
                for result in verification_results:
                    sims = result.get('similarities', [])
                    if i < len(sims):
                        target_sims.append(float(sims[i]))
                
                if target_sims:
                    f.write(f"目标{i+1}:\n")
                    f.write(f"  平均相似度: {np.mean(target_sims):.3f}\n")
                    f.write(f"  相似度标准差: {np.std(target_sims):.3f}\n")
                    f.write(f"  相似度范围: {np.min(target_sims):.3f} - {np.max(target_sims):.3f}\n\n")
        
        # 目标说话人片段详情
        f.write("目标说话人片段详情 (前20个):\n")
        f.write("-" * 40 + "\n")
        for i, seg in enumerate(target_segments[:20]):
            f.write(f"{i+1:3d}. 文件: {os.path.basename(seg.get('file_path', ''))}\n")
            f.write(f"     来源: {seg.get('source_file', 'unknown')}\n")
            f.write(f"     开始时间: {float(seg.get('start_time', 0)):.1f}s\n")
            f.write(f"     结束时间: {float(seg.get('end_time', 0)):.1f}s\n")
            f.write(f"     时长: {float(seg.get('duration', 0)):.1f}s\n")
            f.write(f"     相似度: {float(seg.get('similarity', 0)):.3f}\n\n")
        
        if len(target_segments) > 20:
            f.write(f"... 还有 {len(target_segments) - 20} 个片段\n")
    
    # JSON报告
    json_path = os.path.join(output_dir, "detailed_results.json")
    
    def convert_numpy_types(obj):
        """将NumPy数据类型转换为Python原生类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    simple_results = []
    for result in verification_results:
        # 转换所有NumPy类型
        converted_result = convert_numpy_types({
            'file': os.path.basename(result.get('file_path', '')),
            'source': result.get('source_file', ''),
            'start_time': result.get('start_time', 0),
            'end_time': result.get('end_time', 0),
            'duration': result.get('duration', 0),
            'similarity': result.get('similarity', 0),
            'is_target': result.get('is_target_speaker', False)
        })
        simple_results.append(converted_result)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(simple_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 分析报告已生成:")
    print(f"  - 文本报告: {report_path}")
    print(f"  - JSON详情: {json_path}")

# ==================== 5. 主工作流程 ====================
def main_workflow(input_dir, target_dir, output_base_dir="voice_recognition_results", 
                 threshold=0.6, strategy='mean', show_all_comparisons=False):
    """
    主工作流程
    
    参数：
    - input_dir: 输入音频文件夹
    - target_dir: 目标音频文件夹
    - output_base_dir: 输出基础目录
    - threshold: 相似度阈值
    - strategy: 多目标比对策略 ('mean', 'max', 'vote')
    - show_all_comparisons: 是否显示所有比对结果
    """
    print("=" * 80)
    print("声纹识别完整工作流程 v1.2 - 自动检测所有目标文件")
    print("=" * 80)
    
    # 步骤0: 自动检测并预处理目标音频
    print("\n步骤0: 自动检测并预处理目标音频")
    print("-" * 40)
    
    # 获取目标目录下所有音频文件
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
    target_files = []
    
    if os.path.exists(target_dir):
        for file in os.listdir(target_dir):
            file_path = os.path.join(target_dir, file)
            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in audio_extensions):
                target_files.append(file_path)
    
    if not target_files:
        print(f"❌ 目标目录 '{target_dir}' 中没有找到音频文件")
        return None, None
    
    print(f"✅ 在目标目录中找到 {len(target_files)} 个音频文件:")
    for i, target in enumerate(target_files, 1):
        print(f"  {i:2d}. {os.path.basename(target)}")
    
    # 预处理所有目标音频
    target_prepare_dir = os.path.join(output_base_dir, "prepared_targets")
    valid_targets = prepare_target_audios(target_files, target_prepare_dir)
    
    if not valid_targets:
        print("❌ 没有有效的目标音频，流程终止")
        return None, None
    
    # 步骤1: 音频预处理
    print("\n步骤1: 音频预处理")
    print("-" * 40)
    converted_dir = os.path.join(output_base_dir, "converted_16k")
    converted_files = batch_convert_audio(input_dir, converted_dir)
    
    if not converted_files:
        print("❌ 输入目录中没有找到可处理的音频文件")
        return None, None
    
    # 步骤2: 音频切分
    print("\n步骤2: 音频精细切分")
    print("-" * 40)
    split_dir = os.path.join(output_base_dir, "split_audio")
    all_segments = batch_fine_split(converted_files, split_dir)
    
    if not all_segments:
        print("❌ 没有成功切分出音频片段")
        return None, None
    
    # 步骤3: 初始化声纹模型
    print("\n步骤3: 初始化声纹比对模型")
    print("-" * 40)
    try:
        sv_pipeline = pipeline(
            task='speaker-verification',
            model='iic/speech_eres2net_sv_zh-cn_16k-common'
        )
        print("✅ 模型初始化成功")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return None, None
    
    # 步骤4: 多目标声纹比对
    print("\n步骤4: 多目标声纹比对")
    print("-" * 40)
    verification_results = verify_all_segments_multi(
        all_segments, valid_targets, sv_pipeline, 
        threshold=threshold, strategy=strategy, 
        show_all=show_all_comparisons
    )
    
    # 步骤5: 保存结果
    print("\n步骤5: 保存目标说话人片段")
    print("-" * 40)
    target_speaker_dir = os.path.join(output_base_dir, "target_speaker")
    target_segments = save_target_speaker_segments(verification_results, target_speaker_dir)
    
    # 步骤6: 生成报告
    print("\n步骤6: 生成分析报告")
    print("-" * 40)
    report_dir = os.path.join(output_base_dir, "analysis_reports")
    generate_comprehensive_report(
        verification_results, target_segments, valid_targets, 
        report_dir, strategy, threshold
    )
    
    # 步骤7: 生成统计摘要
    print("\n步骤7: 生成统计摘要")
    print("-" * 40)
    generate_statistical_summary(verification_results, target_segments, target_dir, input_dir, output_base_dir)
    
    print("\n" + "=" * 80)
    print("🎉 工作流程完成！")
    print("=" * 80)
    
    return verification_results, target_segments

def generate_statistical_summary(verification_results, target_segments, target_dir, input_dir, output_base_dir):
    """生成统计摘要"""
    summary_path = os.path.join(output_base_dir, "summary.txt")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("声纹识别系统执行摘要\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("输入配置:\n")
        f.write("-" * 40 + "\n")
        f.write(f"输入音频目录: {input_dir}\n")
        
        # 统计输入目录中的文件
        if os.path.exists(input_dir):
            audio_files = [f for f in os.listdir(input_dir) 
                          if os.path.isfile(os.path.join(input_dir, f)) and 
                          f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'))]
            f.write(f"输入音频文件数: {len(audio_files)}\n")
            f.write("输入文件列表:\n")
            for audio_file in audio_files:
                f.write(f"  - {audio_file}\n")
        f.write("\n")
        
        f.write(f"目标音频目录: {target_dir}\n")
        # 统计目标目录中的文件
        if os.path.exists(target_dir):
            target_files = [f for f in os.listdir(target_dir) 
                           if os.path.isfile(os.path.join(target_dir, f)) and 
                           f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'))]
            f.write(f"目标音频文件数: {len(target_files)}\n")
            f.write("目标文件列表:\n")
            for target_file in target_files:
                f.write(f"  - {target_file}\n")
        f.write("\n")
        
        f.write("处理结果:\n")
        f.write("-" * 40 + "\n")
        f.write(f"总处理片段数: {len(verification_results)}\n")
        f.write(f"识别为目标说话人的片段数: {len(target_segments)}\n")
        f.write(f"目标片段占比: {len(target_segments)/len(verification_results)*100:.1f}%\n")
        
        # 按输入文件统计
        f.write("\n按输入文件统计:\n")
        file_stats = {}
        for result in verification_results:
            source = result.get('source_file', 'unknown')
            is_target = result.get('is_target_speaker', False)
            if source not in file_stats:
                file_stats[source] = {'total': 0, 'target': 0}
            file_stats[source]['total'] += 1
            if is_target:
                file_stats[source]['target'] += 1
        
        for source, stats in file_stats.items():
            f.write(f"  {source}:\n")
            f.write(f"    总片段: {stats['total']} | 目标片段: {stats['target']} | 占比: {stats['target']/stats['total']*100:.1f}%\n")
        
        f.write("\n输出目录结构:\n")
        f.write("-" * 40 + "\n")
        output_structure = [
            f"{output_base_dir}/converted_16k/ - 预处理后的16kHz音频",
            f"{output_base_dir}/split_audio/ - 精细切分后的音频片段",
            f"{output_base_dir}/prepared_targets/ - 预处理后的目标音频",
            f"{output_base_dir}/target_speaker/ - 识别的目标说话人片段",
            f"{output_base_dir}/analysis_reports/ - 详细分析报告"
        ]
        for line in output_structure:
            f.write(f"{line}\n")
    
    print(f"📄 执行摘要已生成: {summary_path}")

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 安装必要依赖
    print("检查依赖安装...")
    required_packages = ['librosa', 'soundfile', 'numpy', 'modelscope']
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"⚠️  {package} 未安装，请运行: pip install {package}")
    
    print("\n" + "=" * 80)
    
    # 定义输入和输出目录
    INPUT_DIR = "input"  # 输入音频目录
    TARGET_DIR = "target"  # 目标音频目录
    OUTPUT_DIR = "voice_recognition_results"  # 输出结果目录
    
    print(f"输入音频目录: {INPUT_DIR}")
    print(f"目标音频目录: {TARGET_DIR}")
    print(f"输出结果目录: {OUTPUT_DIR}")
    print("-" * 40)
    
    # 检查输入目录是否存在
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 输入目录不存在: {INPUT_DIR}")
        print("请创建输入目录并放置音频文件，或修改INPUT_DIR变量")
        exit(1)
    
    if not os.path.exists(TARGET_DIR):
        print(f"❌ 目标目录不存在: {TARGET_DIR}")
        print("请创建目标目录并放置目标说话人样本，或修改TARGET_DIR变量")
        exit(1)
    
    # 运行主工作流程
    results = main_workflow(
        input_dir=INPUT_DIR,
        target_dir=TARGET_DIR,  # 修改为传递目录而不是文件列表
        output_base_dir=OUTPUT_DIR,
        threshold=0.70,   # 调整阈值
        strategy='mean',
        show_all_comparisons=True
    )
    
    if results[0] is not None:
        print(f"\n✅ 处理完成！请查看 {OUTPUT_DIR} 目录中的结果")
    else:
        print("\n❌ 处理失败，请检查错误信息")