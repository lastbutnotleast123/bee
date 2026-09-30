import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = False  # 关闭自动布局，手动控制

def load_e_nose_data_from_multiple_folders(main_data_folder):
    """
    从多个子文件夹读取所有.txt文件并构建数据集
    """
    data_list = []
    labels = []
    file_paths = []
    folder_names = []

    print(f"扫描主文件夹: {main_data_folder}")

    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(main_data_folder)
                  if os.path.isdir(os.path.join(main_data_folder, f))]

    print(f"找到 {len(subfolders)} 个子文件夹: {subfolders}")

    if len(subfolders) == 0:
        print("未找到子文件夹，尝试直接读取主文件夹中的文件...")
        subfolders = ['']

    total_files = 0

    for folder in subfolders:
        folder_path = os.path.join(main_data_folder, folder)

        # 获取该文件夹中的所有.txt文件
        if folder == '':
            txt_files = [f for f in os.listdir(main_data_folder) if f.endswith('.txt')]
            current_folder = "主文件夹"
        else:
            txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
            current_folder = folder

        print(f"\n扫描文件夹 '{current_folder}': 找到 {len(txt_files)} 个.txt文件")

        for file in sorted(txt_files):
            try:
                if folder == '':
                    file_path = os.path.join(main_data_folder, file)
                else:
                    file_path = os.path.join(folder_path, file)

                # 读取文件
                data = np.loadtxt(file_path)

                print(f"文件 {file} 的数据形状: {data.shape}")
                print(f"数据维度: {data.ndim}")

                # 处理二维数据：传感器 × 时间点
                if data.ndim == 2:
                    sensor_count, time_points = data.shape
                    print(f"传感器数量: {sensor_count}, 时间点数: {time_points}")

                    # 提取每个传感器的统计特征
                    features = []
                    feature_names = []

                    for sensor_idx in range(sensor_count):
                        sensor_response = data[sensor_idx, :]

                        # 8个统计特征
                        features.extend([
                            np.mean(sensor_response),  # 平均值
                            np.std(sensor_response),  # 标准差
                            np.max(sensor_response),  # 最大值
                            np.min(sensor_response),  # 最小值
                            np.median(sensor_response),  # 中位数
                            np.percentile(sensor_response, 25),  # 25%分位数
                            np.percentile(sensor_response, 75),  # 75%分位数
                            np.ptp(sensor_response),  # 极差 (max-min)
                        ])

                        feature_names.extend([
                            f'Sensor{sensor_idx}_mean',
                            f'Sensor{sensor_idx}_std',
                            f'Sensor{sensor_idx}_max',
                            f'Sensor{sensor_idx}_min',
                            f'Sensor{sensor_idx}_median',
                            f'Sensor{sensor_idx}_q25',
                            f'Sensor{sensor_idx}_q75',
                            f'Sensor{sensor_idx}_range',
                        ])

                    print(f"  提取特征数量: {len(features)}")

                    data_list.append(features)
                    file_paths.append(file_path)

                    # 使用文件夹名称作为标签
                    if folder == '':
                        label = extract_label_from_filename(file)
                    else:
                        label = folder  # 使用文件夹名作为标签

                    labels.append(label)
                    folder_names.append(current_folder)

                    print(f"  已加载: {file} -> 标签: {label}")
                    total_files += 1

                else:
                    print(f"⚠️ 未知数据维度: {data.ndim}, 跳过此文件")

            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")

    # 转换为DataFrame
    if len(data_list) > 0:
        feature_df = pd.DataFrame(data_list)
        # 设置特征名称
        if len(data_list[0]) == len(feature_names):
            feature_df.columns = feature_names
    else:
        feature_df = pd.DataFrame()

    label_df = pd.Series(labels, name='label')
    folder_df = pd.Series(folder_names, name='folder')

    print(f"\n" + "=" * 60)
    print(f"数据加载完成!")
    print(f"总计成功加载 {total_files} 个样本")
    if len(data_list) > 0:
        print(f"特征维度: {feature_df.shape}")
        print(f"传感器数量: {feature_df.shape[1] // 8}")  # 每个传感器8个特征
        print(f"每个传感器特征数: 8")
    print(f"\n类别分布:")
    print(label_df.value_counts())
    print(f"\n文件夹分布:")
    print(folder_df.value_counts())

    return feature_df, label_df, file_paths

def extract_label_from_filename(filename):
    """
    从文件名提取标签 - 根据你提供的文件名格式
    """
    # 去掉扩展名
    name_without_ext = os.path.splitext(filename)[0]

    # 提取主要类别（去掉数字）
    import re
    # 去掉末尾的数字
    label = re.sub(r'\d+\.?\d*$', '', name_without_ext)

    return label

# 使用示例 - 请修改为你的主文件夹路径
main_data_folder = r"D:\桌面\电子鼻牛大力数据"  # 请修改为实际路径
print("开始加载数据..." + "=" * 50)
feature_df, label_df, file_paths = load_e_nose_data_from_multiple_folders(main_data_folder)

# 如果数据加载成功，继续执行以下代码
if len(feature_df) > 0:
    # 1. 数据基本信息
    print("\n" + "=" * 60)
    print("数据基本信息:")
    print(f"样本数量: {feature_df.shape[0]}")
    print(f"特征数量: {feature_df.shape[1]}")
    sensor_count = feature_df.shape[1] // 8
    features_per_sensor = 8
    print(f"传感器数量: {sensor_count}")
    print(f"类别数量: {len(label_df.unique())}")
    print(f"类别列表: {list(label_df.unique())}")

    # 显示特征结构信息
    feature_names = list(feature_df.columns)

    print(f"\n特征结构:")
    print(f"总特征数: {len(feature_names)}")
    print(f"传感器数量: {sensor_count}")
    print(f"每个传感器特征数: {features_per_sensor}")
    feature_types = ['mean', 'std', 'max', 'min', 'median', 'q25', 'q75', 'range']
    feature_types_cn = ['均值', '标准差', '最大值', '最小值', '中位数', '25分位', '75分位', '极差']
    print(f"特征类型: {feature_types}")

    # 2. 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)
    y = label_df

    # 对标签进行编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"\n标签编码映射:")
    for i, class_name in enumerate(le.classes_):
        print(f"  {class_name} -> {i}")

    # 3. 传感器特征结构可视化（彻底修复所有重叠和截断问题）
    print("\n" + "=" * 60)
    print("传感器特征结构可视化")
    print("=" * 60)

    # 创建超大图形，彻底解决截断和重叠问题
    fig = plt.figure(figsize=(40, 32))
    fig.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.08, 
                       wspace=0.50, hspace=0.60)

    # 手动设置每个子图的精确位置 [left, bottom, width, height]
    # 第一行 - 确保标题完全不被截断
    ax1 = fig.add_axes([0.05, 0.78, 0.26, 0.12])  # 特征组织结构
    ax2 = fig.add_axes([0.37, 0.78, 0.26, 0.12])  # 样本特征值分布
    ax3 = fig.add_axes([0.69, 0.78, 0.26, 0.12])  # 传感器均值热图
    # 第二行 - 确保标题完全不被截断
    ax4 = fig.add_axes([0.05, 0.48, 0.26, 0.12])  # 特征相关性
    ax5 = fig.add_axes([0.37, 0.48, 0.26, 0.12])  # 类别特征分布
    ax6 = fig.add_axes([0.69, 0.48, 0.26, 0.12])  # 特征重要性

    # 1. 特征组织结构热图 - 修复数据异常值和标签重叠
    feature_matrix = np.zeros((sensor_count, features_per_sensor))
    for i in range(sensor_count):
        for j in range(features_per_sensor):
            feature_idx = i * features_per_sensor + j
            feature_values = X_scaled[:, feature_idx]
            # 处理异常值，使用中位数替代极端值
            median_val = np.median(feature_values)
            std_val = np.std(feature_values)
            # 限制在合理范围内
            feature_values = np.clip(feature_values, median_val - 3*std_val, median_val + 3*std_val)
            feature_matrix[i, j] = np.mean(feature_values)

    im1 = ax1.imshow(feature_matrix, cmap='coolwarm', aspect='auto', interpolation='nearest')
    ax1.set_xlabel('特征类型', fontsize=12, labelpad=20)
    ax1.set_ylabel('传感器', fontsize=12, labelpad=20)
    ax1.set_title(f'特征组织结构\n({sensor_count}传感器 × {features_per_sensor}特征)', 
                  fontsize=12, pad=15, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.5, pad=0.10)
    cbar1.set_label('标准化特征值', fontsize=10, labelpad=12)
    
    # 彻底修复X轴标签重叠问题 - 使用极小的字体和极大的间距
    ax1.set_xticks(range(features_per_sensor))
    ax1.set_xticklabels(feature_types_cn, rotation=80, ha='right', fontsize=7)
    ax1.tick_params(axis='x', pad=25)
    
    # 修复Y轴标签
    if sensor_count <= 14:
        ax1.set_yticks(range(sensor_count))
        ax1.set_yticklabels([f'S{i}' for i in range(sensor_count)], fontsize=8)
    else:
        step = max(1, sensor_count // 10)
        y_ticks = range(0, sensor_count, step)
        y_labels = [f'S{i}' for i in y_ticks]
        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels(y_labels, fontsize=7)

    # 2. 单个样本特征值分布 - 修复标签重叠和图例位置
    sample_idx = 0
    sample_features = X_scaled[sample_idx]

    sensors_to_show = min(2, sensor_count)
    x_pos = np.arange(features_per_sensor)
    bar_width = 0.35
    colors = plt.cm.Set2(np.linspace(0, 1, sensors_to_show))

    bars = []
    for sensor_idx in range(sensors_to_show):
        start_idx = sensor_idx * features_per_sensor
        end_idx = start_idx + features_per_sensor
        sensor_features = sample_features[start_idx:end_idx]
        
        bar = ax2.bar(x_pos + sensor_idx * bar_width, sensor_features,
                width=bar_width, color=colors[sensor_idx], alpha=0.8)
        bars.append(bar)

    ax2.set_xlabel('特征类型', fontsize=12, labelpad=20)
    ax2.set_ylabel('标准化特征值', fontsize=12, labelpad=20)
    ax2.set_title(f'样本特征值分布\n(前{sensors_to_show}个传感器)', 
                  fontsize=12, pad=15, fontweight='bold')
    
    # 彻底修复X轴标签重叠 - 使用极小的字体和极大的间距
    ax2.set_xticks(x_pos + bar_width/2)
    ax2.set_xticklabels(feature_types_cn, rotation=80, ha='right', fontsize=7)
    ax2.tick_params(axis='x', pad=25)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 修复图例位置，移到图表外部
    sensor_labels = [f'传感器{i}' for i in range(sensors_to_show)]
    ax2.legend(bars, sensor_labels, loc='center left', fontsize=10, 
              framealpha=0.9, bbox_to_anchor=(1.12, 0.5))

    # 3. 传感器均值热图 - 修复X轴标签
    mean_by_sensor = np.zeros((len(X_scaled), sensor_count))
    for i in range(sensor_count):
        mean_by_sensor[:, i] = X_scaled[:, i * features_per_sensor]

    if len(mean_by_sensor) > 30:
        step = max(1, len(mean_by_sensor) // 30)
        display_data = mean_by_sensor[::step].T
        x_labels = [f'样本{i*step}' for i in range(0, len(mean_by_sensor), step)]
    else:
        display_data = mean_by_sensor.T
        x_labels = [f'样本{i}' for i in range(len(mean_by_sensor))]

    im3 = ax3.imshow(display_data, cmap='RdYlBu', aspect='auto', interpolation='nearest')
    ax3.set_xlabel('样本索引', fontsize=12, labelpad=20)
    ax3.set_ylabel('传感器', fontsize=12, labelpad=20)
    ax3.set_title(f'传感器均值特征热图\n({sensor_count}个传感器)', 
                  fontsize=12, pad=15, fontweight='bold')
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.5, pad=0.10)
    cbar3.set_label('标准化均值', fontsize=10, labelpad=12)
    
    # 修复X轴标签 - 减少标签数量，避免重叠
    if len(x_labels) <= 5:
        ax3.set_xticks(range(len(x_labels)))
        ax3.set_xticklabels(x_labels, rotation=80, ha='right', fontsize=7)
    else:
        step = max(1, len(x_labels) // 4)  # 只显示4个标签
        x_ticks = range(0, len(x_labels), step)
        ax3.set_xticks(x_ticks)
        ax3.set_xticklabels([x_labels[i] for i in x_ticks], rotation=80, ha='right', fontsize=6)
    ax3.tick_params(axis='x', pad=25)
    
    # 修复Y轴标签
    if sensor_count <= 14:
        ax3.set_yticks(range(sensor_count))
        ax3.set_yticklabels([f'S{i}' for i in range(sensor_count)], fontsize=8)
    else:
        step = max(1, sensor_count // 10)
        y_ticks = range(0, sensor_count, step)
        y_labels = [f'S{i}' for i in y_ticks]
        ax3.set_yticks(y_ticks)
        ax3.set_yticklabels(y_labels, fontsize=7)

    # 4. 特征相关性热图 - 修复标题和标签
    corr_sensors = min(5, sensor_count)
    selected_features = [i * features_per_sensor for i in range(corr_sensors)]
    correlation_data = X_scaled[:, selected_features]
    corr_matrix = np.corrcoef(correlation_data.T)

    im4 = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax4.set_xlabel('传感器', fontsize=12, labelpad=20)
    ax4.set_ylabel('传感器', fontsize=12, labelpad=20)
    ax4.set_title(f'传感器均值特征相关性\n(前{corr_sensors}个传感器)', 
                  fontsize=12, pad=15, fontweight='bold')
    cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.5, pad=0.10)
    cbar4.set_label('相关系数', fontsize=10, labelpad=12)
    ax4.set_xticks(range(corr_sensors))
    ax4.set_yticks(range(corr_sensors))
    ax4.set_xticklabels([f'S{i}' for i in range(corr_sensors)], fontsize=9)
    ax4.set_yticklabels([f'S{i}' for i in range(corr_sensors)], fontsize=9)

    # 5. 类别特征分布箱线图 - 修复标签截断和重叠
    sensor_idx = 0
    feature_idx = sensor_idx * features_per_sensor

    boxplot_data = []
    boxplot_labels = []
    for class_idx, class_name in enumerate(le.classes_):
        class_mask = (y_encoded == class_idx)
        if np.sum(class_mask) > 0:
            boxplot_data.append(X_scaled[class_mask, feature_idx])
            # 截断过长的标签，避免重叠
            if len(class_name) > 8:
                display_name = class_name[:8] + '...'
            else:
                display_name = class_name
            boxplot_labels.append(display_name)

    if boxplot_data:
        box_plot = ax5.boxplot(boxplot_data, tick_labels=boxplot_labels, 
                              patch_artist=True, showmeans=True)
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(boxplot_data)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax5.set_ylabel('标准化特征值', fontsize=12, labelpad=20)
        ax5.set_title(f'传感器0均值特征分布', fontsize=12, pad=15, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 修复X轴标签，确保完整显示 - 使用极小的字体和极大的间距
        ax5.set_xticklabels(boxplot_labels, rotation=80, ha='right', fontsize=6)
        # 增加底部边距，确保标签不被截断
        ax5.tick_params(axis='x', pad=30)

    # 6. 特征重要性 - 修复标签显示
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y_encoded)
        importances = rf.feature_importances_
        
        top_n = min(6, len(importances))
        top_indices = np.argsort(importances)[-top_n:][::-1]
        top_importances = importances[top_indices]
        top_features = [feature_df.columns[i] for i in top_indices]
        
        simplified_names = []
        for name in top_features:
            parts = name.split('_')
            if len(parts) >= 2:
                feature_idx = int(parts[0].replace('Sensor', ''))
                feature_type = parts[1]
                # 使用简短但可识别的特征名称
                feature_short = {
                    'mean': '均值', 'std': '标准差', 'max': '最大', 
                    'min': '最小', 'median': '中位数', 'q25': '25分位',
                    'q75': '75分位', 'range': '极差'
                }.get(feature_type, feature_type[:3])
                simplified_names.append(f'S{feature_idx}\n{feature_short}')
            else:
                simplified_names.append(name[:6])
        
        y_pos = np.arange(len(top_importances))
        ax6.barh(y_pos, top_importances, color='lightblue', alpha=0.8, height=0.6)
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(simplified_names, fontsize=12)
        ax6.set_xlabel('特征重要性', fontsize=12, labelpad=20)
        ax6.set_title('Top 6 重要特征', fontsize=12, pad=15, fontweight='bold')
        ax6.invert_yaxis()
        ax6.grid(True, alpha=0.3, axis='x')
        
        for i, v in enumerate(top_importances):
            ax6.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=11)

    except Exception as e:
        ax6.text(0.5, 0.5, '特征重要性\n分析失败', ha='center', va='center', 
                transform=ax6.transAxes, fontsize=12)
        ax6.set_title('特征重要性分析', fontsize=16, fontweight='bold')

    plt.show()

    # 4. 样本特征值分布可视化（修复第二张图的布局和标签问题）
    print("\n" + "=" * 60)
    print("样本特征值分布可视化")
    print("=" * 60)

    unique_labels = le.classes_
    n_labels = min(3, len(unique_labels))

    # 创建第二张图，彻底解决截断和空间配比问题
    fig2 = plt.figure(figsize=(32, 10 * n_labels))
    fig2.subplots_adjust(left=0.10, right=0.90, top=0.94, bottom=0.12, hspace=0.5)

    for label_idx, label in enumerate(unique_labels[:n_labels]):
        label_indices = np.where(y_encoded == label_idx)[0]
        
        if len(label_indices) == 0:
            continue
            
        data_idx = label_indices[0]
        # 手动设置每个子图位置，彻底优化空间配比
        ax = fig2.add_axes([0.10, 0.88 - label_idx*0.28, 0.80, 0.22])
        
        sensors_to_show = min(2, sensor_count)
        feature_indices = [0, 1, 2, 3]  # 选择前4个特征
        
        bar_data = []
        for sensor_idx in range(sensors_to_show):
            sensor_features = []
            for feat_idx in feature_indices:
                feature_idx = sensor_idx * features_per_sensor + feat_idx
                sensor_features.append(X_scaled[data_idx, feature_idx])
            bar_data.append(sensor_features)
        
        bar_data = np.array(bar_data)
        
        y_pos = np.arange(len(feature_indices))
        bar_height = 0.3
        colors = plt.cm.tab10(np.linspace(0, 1, sensors_to_show))
        
        bars = []
        for i in range(sensors_to_show):
            bar = ax.barh(y_pos + i * bar_height, bar_data[i], height=bar_height,
                    color=colors[i], alpha=0.8)
            bars.append(bar)
        
        # 修复Y轴标签，确保不被截断
        ax.set_ylabel('特征类型', fontsize=12, labelpad=25)
        ax.set_xlabel('标准化特征值', fontsize=12, labelpad=20)
        
        # 截断过长的标题，但不要过度截断
        if len(label) > 20:
            display_title = label[:20] + '...'
        else:
            display_title = label
        ax.set_title(f'{display_title} - 样本特征分布', fontsize=12, pad=15, fontweight='bold')
        
        ax.set_yticks(y_pos + bar_height/2)
        ax.set_yticklabels([feature_types_cn[i] for i in feature_indices], fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 修复图例位置，移到图表外部
        sensor_labels = [f'传感器{i}' for i in range(sensors_to_show)]
        ax.legend(bars, sensor_labels, loc='center left', bbox_to_anchor=(1.08, 0.5), 
                 fontsize=10, framealpha=0.9)

    plt.show()

    # 5. PCA分析
    print("\n" + "=" * 60)
    print("进行PCA分析...")
    print("=" * 60)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # 第三张图：PCA分析 - 等比缩小，避免被遮挡
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.subplots_adjust(left=0.12, right=0.88, top=0.90, bottom=0.20, wspace=0.35)

    axes[0].plot(range(1, len(pca.explained_variance_ratio_) + 1),
                 pca.explained_variance_ratio_.cumsum(), 'bo-', linewidth=2, markersize=6)
    axes[0].set_xlabel('主成分数量', fontsize=16, labelpad=20)
    axes[0].set_ylabel('累积方差解释率', fontsize=16, labelpad=20)
    axes[0].set_title('PCA方差解释率', fontsize=16, fontweight='bold', pad=20)
    axes[0].grid(True, alpha=0.3)

    scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='viridis',
                              alpha=0.7, s=60, edgecolors='w', linewidth=0.5)
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=16, labelpad=20)
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=16, labelpad=20)
    axes[1].set_title('PCA 二维散点图', fontsize=16, fontweight='bold', pad=20)

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=plt.cm.viridis(i / (len(le.classes_) - 1)),
                              markersize=8, label=le.classes_[i])
                       for i in range(len(le.classes_))]
    axes[1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

    plt.tight_layout()
    plt.show()

    # 6. LDA分析
    print("\n" + "=" * 60)
    print("进行LDA分析...")
    print("=" * 60)

    try:
        lda = LinearDiscriminantAnalysis()
        X_lda = lda.fit_transform(X_scaled, y_encoded)

        # 第四张图：LDA分析 - 等比缩小，避免被遮挡
        plt.figure(figsize=(12, 10))
        plt.subplots_adjust(left=0.12, right=0.80, top=0.90, bottom=0.20)

        if X_lda.shape[1] == 1:
            for i, class_name in enumerate(le.classes_):
                class_mask = (y_encoded == i)
                plt.scatter(X_lda[class_mask, 0], np.zeros(np.sum(class_mask)),
                            label=class_name, alpha=0.7, s=60, edgecolors='w', linewidth=0.5)
            plt.xlabel('LD1', fontsize=16, labelpad=20)
            plt.ylabel('', fontsize=16)
            plt.title('LDA分析 (一维)', fontsize=16, fontweight='bold', pad=20)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
            plt.grid(True, alpha=0.3)
        else:
            scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y_encoded, cmap='plasma',
                                  alpha=0.7, s=60, edgecolors='w', linewidth=0.5)
            plt.xlabel('LD1', fontsize=16, labelpad=20)
            plt.ylabel('LD2', fontsize=16, labelpad=20)
            plt.title('LDA分析', fontsize=16, fontweight='bold', pad=20)
            
            legend_elements = [Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=plt.cm.plasma(i / (len(le.classes_) - 1)),
                                      markersize=8, label=le.classes_[i])
                               for i in range(len(le.classes_))]
            plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"LDA分析出错: {e}")

    # 7. 分类模型比较
    print("\n" + "=" * 60)
    print("开始训练分类模型...")
    print("=" * 60)

    models = {
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'LDA': LinearDiscriminantAnalysis(),
        '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
        '神经网络': MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
    }

    n_splits = min(5, len(np.unique(y_encoded)), min([np.sum(y_encoded == i) for i in np.unique(y_encoded)]))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"使用 {n_splits} 折交叉验证")

    results = {}

    for name, model in models.items():
        print(f"\n训练 {name} 模型...")

        try:
            y_pred = cross_val_predict(model, X_scaled, y_encoded, cv=cv)
            acc = accuracy_score(y_encoded, y_pred)
            results[name] = acc

            print(f"{name} 准确率: {acc:.4f}")

        except Exception as e:
            print(f"训练 {name} 模型时出错: {e}")
            results[name] = 0

    # 绘制模型性能比较图
    print("\n" + "=" * 60)
    print("模型性能对比")
    print("=" * 60)

    # 第五张图：模型性能比较 - 等比缩小
    plt.figure(figsize=(14, 10))
    plt.subplots_adjust(left=0.20, right=0.88, top=0.90, bottom=0.20)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    model_names = [item[0] for item in sorted_results]
    accuracies = [item[1] for item in sorted_results]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    bars = plt.barh(range(len(model_names)), accuracies, color=colors, alpha=0.8)
    
    plt.xlabel('准确率', fontsize=16, labelpad=20)
    plt.ylabel('模型', fontsize=16, labelpad=20)
    plt.title('分类模型性能比较', fontsize=16, fontweight='bold', pad=20)
    plt.yticks(range(len(model_names)), model_names, fontsize=13)
    plt.xlim(0, 1.0)
    plt.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{acc:.4f}', ha='left', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

    print("-" * 40)
    for name, acc in sorted_results:
        print(f"{name:10} : {acc:.4f}")

    if results and max(results.values()) > 0:
        best_model_name = max(results, key=results.get)
        best_accuracy = results[best_model_name]
        print(f"\n🎯 最佳模型: {best_model_name} (准确率: {best_accuracy:.4f})")

    print("\n✨ 分析完成！")

else:
    print("没有成功加载数据，请检查主文件夹路径")
