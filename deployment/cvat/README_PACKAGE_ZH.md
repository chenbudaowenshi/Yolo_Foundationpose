# CVAT 分割标注平台安装包

这个包用来在本机用 Docker 部署 CVAT v2.70.0，在网页里修正 YOLO 分割标签，然后导回 YOLOv8 数据集。安装包本身只有几百 KB，Docker 镜像在安装时联网拉取，首次安装约 3 GB。

## 环境要求

- Linux（Ubuntu 22.04 已验证）。Windows/macOS 也可以用 Docker Desktop 部署，但 `install.sh` 只适用于 Linux；
- Docker Engine 和 Docker Compose v2.24.4 以上版本；
- 不需要 GPU；
- 本机 8080 端口未被其他程序占用。

## 一键安装

```bash
sha256sum -c cvat-annotation-bundle.tar.gz.sha256
tar -xzf cvat-annotation-bundle.tar.gz
cd cvat_annotation_bundle

./install.sh --dataset /path/to/your_yolo_dataset
```

脚本会依次检查 Docker、生成配置、拉取镜像、启动 CVAT、等待服务就绪，最后询问是否创建管理员账号。安装完成后用浏览器打开 <http://localhost:8080>，用刚创建的账号登录。

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--dataset DIR` | 把数据集以只读方式挂进 CVAT，可重复使用，在 CVAT 中显示为 `/<目录名>` |
| `--install-docker` | 机器上还没有 Docker 时，用 apt 自动安装（需要 sudo）。装完后重新登录一次再运行 `install.sh` |
| `--mirror dockerproxy.net` | 连不上 Docker Hub 时，通过镜像源拉取 |
| `--lan 192.168.x.x` | 让局域网内其他人通过 `http://192.168.x.x:8080` 访问。默认只允许本机访问 |
| `--admin` / `--no-admin` | 直接创建管理员 / 跳过创建，不再询问 |

`install.sh` 可以重复运行，例如换数据集或改成局域网访问时，带上新参数再跑一次即可。已有的任务和标注不会丢失。

如果提示 `cannot talk to the Docker daemon`，说明当前用户还不在 `docker` 组中：

```bash
sudo usermod -aG docker $USER
# 注销后重新登录，或者在当前终端执行：
newgrp docker
```

## 标注流程

### 1. 把已有的 YOLO 标签打包成 CVAT 可导入的格式

需要 Python 3 和 PyYAML（`pip install pyyaml`）。数据集应为 YOLOv8 分割格式，即包含 `data.yaml`、`images/{train,val}` 和 `labels/{train,val}`。

```bash
python tools/prepare_cvat_segmentation.py \
  --dataset-root /path/to/your_yolo_dataset \
  --output-dir ./cvat_import
```

每个 split 会生成一个 ZIP。这个 ZIP 里同时打包了图片，因为 CVAT 2.70 导入时要从压缩包中读取图片尺寸。

### 2. 在 CVAT 中建任务并导入

1. 新建 Project，按 `data.yaml` 中的类别名逐个添加 Label，类型选 Polygon；
2. 在 Project 下新建 Task，选择数据时切到 **Connected file share**，勾选 `/<数据集目录名>/images/train`（val 另建一个 Task）；
3. 进入 Task，点击 **Actions → Upload annotations**，格式选 **Ultralytics YOLO Segmentation 1.0**，上传第 1 步生成的对应 ZIP；
4. 在网页里修正多边形。

### 3. 导出并合并回数据集

在每个 Task 中点击 **Actions → Export task dataset**，格式选 **Ultralytics YOLO Segmentation 1.0**，不要勾选 Save images，下载 train 和 val 两个 ZIP。

先只做校验，不改动数据：

```bash
python tools/import_cvat_segmentation.py \
  --train-zip train_export.zip \
  --val-zip val_export.zip \
  --dataset-root /path/to/your_yolo_dataset
```

确认输出没有问题后，加上 `--apply` 正式写入。脚本会先备份原有标签，并按类别名重新对齐类别 ID，因为 CVAT 在不同 Task 中给出的类别顺序可能不同。

## 日常管理

在 `cvat/` 目录下执行：

```bash
docker compose ps                    # 查看状态
docker compose stop                  # 停止（数据保留）
docker compose start                 # 再次启动
docker compose logs -f cvat_server   # 查看日志
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'  # 新增管理员
```

所有任务和标注都保存在 Docker volume 中，`docker compose down` 也不会删除这些数据。

**注意：**`docker compose down -v` 会删除所有任务和标注，执行前请先把需要的标注导出。

## 许可证

CVAT 以 MIT 许可证发布，见 `cvat/LICENSE-CVAT-MIT.txt`。
