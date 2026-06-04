##  Инструкция по запуску

### Скачивание весов модели  **[ссылка (Google Drive)](https://drive.google.com/file/d/1TQFyHny_h6xkheqRtX_aOhzc1o51jDbh/view?usp=sharing)**

###  Сборка Docker-образа
Откройте терминал в папке с репозиторием и выполните команду:
```bash
docker build -t action_model .
```

###  Запуск инференса (Тестирование)
Чтобы протестировать модель на, примонтируйте директорию с данными внутрь контейнера (замените /путь/к/тестовой/папке на ваш абсолютный путь)

**Для Linux / Mac:**
```bash
docker run --rm -v /путь/к/тестовой/папке:/data action_model /data
```

**Для Windows (CMD):**
```Cmd
docker run --rm -v "C:\путь\к\тестовой\папке":/data action_model /data
```
