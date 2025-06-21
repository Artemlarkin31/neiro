import logging
import asyncio
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, FSInputFile
from aiogram.utils.markdown import text
from config import Config  # Для хранения токена
import os
from aiogram import F
import time
from aiogram import types
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
import open3d as o3d
import numpy as np
from skimage import measure

# Настройка логов
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация бота
bot = Bot(token=Config.BOT_TOKEN)
dp = Dispatcher()


# ===== МОДЕЛЬ ДЛЯ ГЕНЕРАЦИИ 3D МОДЕЛЕЙ =====
# Классы модели PointCloudNet
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(weights="IMAGENET1K_V1")
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=18,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        self.resnet.fc = nn.Linear(512, 512)
    
    def forward(self, x):
        return self.resnet(x)

class Decoder(nn.Module):
    def __init__(self, num_points=4750):
        super().__init__()
        self.num_points = num_points
        self.fc = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),  # Добавлен новый слой
            nn.ReLU(),
            nn.Linear(2048, num_points * 3),  # Финальный слой теперь 4-й по индексу
            nn.Tanh()
        )
    
    def forward(self, x):
        points = self.fc(x).view(-1, self.num_points, 3)
        return points

class PointCloudNet(nn.Module):
    def __init__(self, num_points=4750):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_points=num_points)
    
    def forward(self, x):
        features = self.encoder(x)
        points = self.decoder(features)
        return points

# ===== ОБРАБОТЧИКИ КОМАНД =====
@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    welcome_text = text(
        f"Привет, {message.from_user.full_name}!",
        "",
        "Этот бот позволяет преобразовывать 3 изображения объекта в 3D-модель следующего формата:",
        "- ogj ",
        "",
        "Для работы с ботом используйте следующие команды:",
        "/generate - Запускает генерацию 3D-модели",
        "/front - вместе с этой командой необходимо прикрепить изображение объекта спереди",
        "/back - вместе с этой командой необходимо прикрепить изображение объекта сзади",
        "/left- вместе с этой командой необходимо прикрепить изображение объекта сбоку слева",
        "/right - вместе с этой командой необходимо прикрепить изображение объекта справа",
        "/top - вместе с этой командой необходимо прикрепить изображение объекта сверху",
        "/bottom - вместе с этой командой необходимо прикрепить изображение объекта снизу",
        "",
        "Отправьте фотографии вашего объекта с 6-х сторон (вид спереди, сзади, сверху, снизу, справа и слева) для создания 3D-модели.",
        sep="\n"
    )
    await message.answer(welcome_text)


# ===== ОБРАБОТЧИКИ ЗАГРУЗКИ ИЗОБРАЖЕНИЙ =====
@dp.message(Command("front"))
async def handle_user_photo_front(message: types.Message):
    try:
        # Получаем данные пользователя
        user_id = str(message.from_user.id)
        photo = message.photo[-1]  # Берем фото максимального качества
        
        # Создаем структуру папок
        user_folder = os.path.join("database", user_id)
        os.makedirs(user_folder, exist_ok=True)  # Создаем папку если не существует
        
        # Формируем путь для сохранения
        photo_path = os.path.join(user_folder, "front.png")
        
        # Скачиваем временный файл
        file = await bot.get_file(photo.file_id)
        temp_path = f"temp_{photo.file_id}.jpg"
        await bot.download_file(file.file_path, temp_path)
        
        # Конвертируем в PNG и сохраняем
        with Image.open(temp_path) as img:
            img.save(photo_path, "PNG")
        
        await message.reply("✅ Фото успешно сохранено как 'front.png'!")
        
    except Exception as e:
        await message.answer(f"❌ Ошибка при обработке фото: {str(e)}")
        
    finally:
        # Удаляем временный файл, если он существует
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
@dp.message(Command("back"))
async def handle_user_photo_back(message: types.Message):
    try:
        # Получаем данные пользователя
        user_id = str(message.from_user.id)
        photo = message.photo[-1]  # Берем фото максимального качества
        
        # Создаем структуру папок
        user_folder = os.path.join("database", user_id)
        os.makedirs(user_folder, exist_ok=True)  # Создаем папку если не существует
        
        # Формируем путь для сохранения
        photo_path = os.path.join(user_folder, "back.png")
        
        # Скачиваем временный файл
        file = await bot.get_file(photo.file_id)
        temp_path = f"temp_{photo.file_id}.jpg"
        await bot.download_file(file.file_path, temp_path)
        
        # Конвертируем в PNG и сохраняем
        with Image.open(temp_path) as img:
            img.save(photo_path, "PNG")
        
        await message.reply("✅ Фото успешно сохранено как 'back.png'!")
        
    except Exception as e:
        await message.answer(f"❌ Ошибка при обработке фото: {str(e)}")
        
    finally:
        # Удаляем временный файл, если он существует
        if os.path.exists(temp_path):
            os.remove(temp_path)

@dp.message(Command("left"))
async def handle_user_photo_left(message: types.Message):
    try:
        # Получаем данные пользователя
        user_id = str(message.from_user.id)
        photo = message.photo[-1]  # Берем фото максимального качества
        
        # Создаем структуру папок
        user_folder = os.path.join("database", user_id)
        os.makedirs(user_folder, exist_ok=True)  # Создаем папку если не существует
        
        # Формируем путь для сохранения
        photo_path = os.path.join(user_folder, "left.png")
        
        # Скачиваем временный файл
        file = await bot.get_file(photo.file_id)
        temp_path = f"temp_{photo.file_id}.jpg"
        await bot.download_file(file.file_path, temp_path)
        
        # Конвертируем в PNG и сохраняем
        with Image.open(temp_path) as img:
            img.save(photo_path, "PNG")
        
        await message.reply("✅ Фото успешно сохранено как 'left.png'!")
        
    except Exception as e:
        await message.answer(f"❌ Ошибка при обработке фото: {str(e)}")
        
    finally:
        # Удаляем временный файл, если он существует
        if os.path.exists(temp_path):
            os.remove(temp_path)

@dp.message(Command("right"))
async def handle_user_photo_right(message: types.Message):
    try:
        # Получаем данные пользователя
        user_id = str(message.from_user.id)
        photo = message.photo[-1]  # Берем фото максимального качества
        
        # Создаем структуру папок
        user_folder = os.path.join("database", user_id)
        os.makedirs(user_folder, exist_ok=True)  # Создаем папку если не существует
        
        # Формируем путь для сохранения
        photo_path = os.path.join(user_folder, "right.png")
        
        # Скачиваем временный файл
        file = await bot.get_file(photo.file_id)
        temp_path = f"temp_{photo.file_id}.jpg"
        await bot.download_file(file.file_path, temp_path)
        
        # Конвертируем в PNG и сохраняем
        with Image.open(temp_path) as img:
            img.save(photo_path, "PNG")
        
        await message.reply("✅ Фото успешно сохранено как 'right.png'!")
        
    except Exception as e:
        await message.answer(f"❌ Ошибка при обработке фото: {str(e)}")
        
    finally:
        # Удаляем временный файл, если он существует
        if os.path.exists(temp_path):
            os.remove(temp_path)

@dp.message(Command("top"))
async def handle_user_photo_top(message: types.Message):
    try:
        # Получаем данные пользователя
        user_id = str(message.from_user.id)
        photo = message.photo[-1]  # Берем фото максимального качества
        
        # Создаем структуру папок
        user_folder = os.path.join("database", user_id)
        os.makedirs(user_folder, exist_ok=True)  # Создаем папку если не существует
        
        # Формируем путь для сохранения
        photo_path = os.path.join(user_folder, "top.png")
        
        # Скачиваем временный файл
        file = await bot.get_file(photo.file_id)
        temp_path = f"temp_{photo.file_id}.jpg"
        await bot.download_file(file.file_path, temp_path)
        
        # Конвертируем в PNG и сохраняем
        with Image.open(temp_path) as img:
            img.save(photo_path, "PNG")
        
        await message.reply("✅ Фото успешно сохранено как 'top.png'!")
        
    except Exception as e:
        await message.answer(f"❌ Ошибка при обработке фото: {str(e)}")
        
    finally:
        # Удаляем временный файл, если он существует
        if os.path.exists(temp_path):
            os.remove(temp_path)

@dp.message(Command("bottom"))
async def handle_user_photo_bottom(message: types.Message):
    try:
        # Получаем данные пользователя
        user_id = str(message.from_user.id)
        photo = message.photo[-1]  # Берем фото максимального качества
        
        # Создаем структуру папок
        user_folder = os.path.join("database", user_id)
        os.makedirs(user_folder, exist_ok=True)  # Создаем папку если не существует
        
        # Формируем путь для сохранения
        photo_path = os.path.join(user_folder, "bottom.png")
        
        # Скачиваем временный файл
        file = await bot.get_file(photo.file_id)
        temp_path = f"temp_{photo.file_id}.jpg"
        await bot.download_file(file.file_path, temp_path)
        
        # Конвертируем в PNG и сохраняем
        with Image.open(temp_path) as img:
            img.save(photo_path, "PNG")
        
        await message.reply("✅ Фото успешно сохранено как 'bottom.png'!")
        
    except Exception as e:
        await message.answer(f"❌ Ошибка при обработке фото: {str(e)}")
        
    finally:
        # Удаляем временный файл, если он существует
        if os.path.exists(temp_path):
            os.remove(temp_path)



# ===== ФУНКЦИИ ДЛЯ ГЕНЕРАЦИИ 3D МОДЕЛЕЙ =====
def points_to_obj(points, output_obj):
    # Проверяем форму массива
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Ожидается массив формы (num_points, 3), но получена форма {points.shape}")
    
    # Проверяем на NaN или бесконечные значения
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        raise ValueError("Точечное облако содержит NaN или бесконечные значения")
    
    # Создаём точечное облако в Open3D
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    # Проверяем наличие точек
    if len(point_cloud.points) == 0:
        raise ValueError("Точечное облако пустое, не удалось создать .obj файл")
    
    # Оцениваем нормали (необходимы для реконструкции)
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    
    # Попытка Poisson Surface Reconstruction
    try:
        # Выполняем Poisson Reconstruction
        poisson_result = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud, depth=8, scale=1.1, linear_fit=True
        )
        
        # Проверяем, возвращает ли функция кортеж (mesh, density) или только mesh
        if isinstance(poisson_result, tuple):
            mesh, density = poisson_result
            # Удаляем вершины с низкой плотностью
            density = np.asarray(density)
            vertices_to_remove = density < np.quantile(density, 0.01)
            mesh.remove_vertices_by_mask(vertices_to_remove)
        else:
            mesh = poisson_result
        
        # Проверяем, что сетка валидна
        if not mesh.has_triangles():
            raise ValueError("Poisson Reconstruction не создала треугольников")
        
        print(f"Poisson mesh generated: vertices={len(mesh.vertices)}, triangles={len(mesh.triangles)}")
        
        # Сохраняем сетку как .obj
        success = o3d.io.write_triangle_mesh(output_obj, mesh, write_ascii=True)
        if not success:
            raise ValueError("Open3D не смог записать .obj файл")
    
    except Exception as e:
        print(f"Poisson reconstruction failed: {str(e)}, trying Ball Pivoting Algorithm")
        # Альтернатива: Ball Pivoting Algorithm
        try:
            radii = [0.005, 0.01, 0.02, 0.04]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                point_cloud, o3d.utility.DoubleVector(radii)
            )
            
            # Проверяем, что сетка валидна
            if not mesh.has_triangles():
                raise ValueError("Ball Pivoting Algorithm не создал треугольников")
            
            print(f"Ball Pivoting mesh generated: vertices={len(mesh.vertices)}, triangles={len(mesh.triangles)}")
            
            # Сохраняем сетку как .obj
            success = o3d.io.write_triangle_mesh(output_obj, mesh, write_ascii=True)
            if not success:
                raise ValueError("Open3D не смог записать .obj файл")
        
        except Exception as e2:
            print(f"Ball Pivoting reconstruction failed: {str(e2)}, falling back to point cloud")
            # Если обе реконструкции не удались, сохраняем только точки
            with open(output_obj, 'w') as f:
                for point in points:
                    f.write(f"v {point[0]} {point[1]} {point[2]}\n")
            return
    
    # Проверяем, что файл создан и не пуст
    if not os.path.exists(output_obj):
        raise ValueError(f"Файл {output_obj} не был создан")
    if os.path.getsize(output_obj) == 0:
        raise ValueError(f"Файл {output_obj} пуст")
    
    print(f"Final output: {output_obj}, size: {os.path.getsize(output_obj)} bytes")
    
    
    # Сохраняем сетку как .obj
    success = o3d.io.write_triangle_mesh(output_obj, mesh, write_ascii=True)
    if not success:
        print(f"Open3D failed to write {output_obj}, writing manually")
        # Альтернативная ручная запись .obj (вершины и треугольники)
        with open(output_obj, 'w') as f:
            # Записываем вершины
            for vertex in np.asarray(mesh.vertices):
                f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
            # Записываем треугольники (индексы начинаются с 1 в .obj)
            for triangle in np.asarray(mesh.triangles):
                f.write(f"f {triangle[0]+1} {triangle[1]+1} {triangle[2]+1}\n")
    
    # Проверяем, что файл создан и не пуст
    if not os.path.exists(output_obj):
        raise ValueError(f"Файл {output_obj} не был создан")
    if os.path.getsize(output_obj) == 0:
        raise ValueError(f"Файл {output_obj} пуст")
    
    print(f"Final output: {output_obj}, size: {os.path.getsize(output_obj)} bytes")

# ===== ОБРАБОТЧИК ГЕНЕРАЦИИ МОДЕЛИ =====
@dp.message(Command("generate"))
async def generate_model(message: types.Message):
    try:
        # Получаем ID пользователя
        user_id = str(message.from_user.id)
        user_folder = os.path.join("database", user_id)
        
        if not os.path.exists(user_folder):
            await message.answer("❌ Вы ещё не отправили фото для генерации. Используйте команды /front, /back, /left, /right, /top.")
            return
        
        # Проверяем наличие всех изображений
        required_files = ["front.png", "back.png", "left.png", "right.png", "top.png", "bottom.png"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(user_folder, f))]
        
        if missing_files:
            await message.answer(f"❌ Не хватает изображений: {', '.join(missing_files)}. Загрузите их с помощью /front, /back, /top.")
            return
        
        await message.answer("⏳ Создаю 3D-модель, пожалуйста, подождите...")
        
        # Загрузка и обработка изображений
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        images = []
        for img_file in required_files:
            img_path = os.path.join(user_folder, img_file)
            img = Image.open(img_path).convert("RGB")
            images.append(transform(img))
        
        image_stack = torch.cat(images, dim=0).unsqueeze(0)
        
        # Загрузка модели и генерация точек
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PointCloudNet(num_points=4750).to(device)
        model.load_state_dict(torch.load("PointCloudNet4.pth", map_location=device), strict=False)
        model.eval()
        
        with torch.no_grad():
            points_pred = model(image_stack.to(device)).squeeze(0).cpu().numpy()
        
        # Обработка облака точек
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_pred)
        
        # 1. Удаление выбросов и нормализация
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        points = np.asarray(pcd.points)
        points -= np.mean(points, axis=0)
        points /= np.max(np.linalg.norm(points, axis=1))
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 2. Расчет нормалей
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.17, max_nn=60)
        )
        pcd.orient_normals_consistent_tangent_plane(k=50)
        
        # 3. Poisson реконструкция с оптимальными параметрами
        output_obj = os.path.join(user_folder, "model.obj")
        
        # Параметры Poisson реконструкции:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, 
            depth=12,       # 9 было стало 12 (лучший баланс)
            scale=1.1,      # 1.2 было Слегка уменьшен для уменьшения артефактов
            linear_fit=False,
            n_threads=-1    # Использовать все ядра
        )

        # Адаптивное удаление шума
        density_threshold = np.quantile(densities, 0.03)  # Более мягкий порог
        mesh.remove_vertices_by_mask(densities < density_threshold)
        
        # Постобработка сетки
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        # Сохранение результата
        o3d.io.write_triangle_mesh(output_obj, mesh, write_ascii=True)
        
        # Отправка результата
        if os.path.exists(output_obj) and os.path.getsize(output_obj) > 0:
            await message.answer_document(
                document=types.FSInputFile(output_obj, filename="model.obj"),
                caption="✅ 3D-модель создана\n\n"
            )
        else:
            await message.answer("❌ Не удалось создать модель.")
        
    except Exception as e:
        error_msg = f"❌ Ошибка при создании модели: {str(e)}"
        print(f"Error in /generate: {error_msg}")
        await message.answer(error_msg)
# ===== ЗАПУСК БОТА =====
async def main():
    # logger.info("Starting bot") стоит удалить после выпуска бота в итоговом проекте
    # , так как сервис будет долго выдавать в консоль действия множества пользователей.
    logger.info("Starting bot")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Exit')