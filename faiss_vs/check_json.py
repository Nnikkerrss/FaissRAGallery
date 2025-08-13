#!/usr/bin/env python3
import json

# Проверяем JSON файл
json_file = "faiss/clients/6a2502fa-caaa-11e3-9af3-e41f13beb1d2/data.json"

try:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"🔍 Проверка JSON файла: {json_file}")
    print("=" * 60)

    if 'result' in data:
        result = data['result']
        print(f"📊 Количество документов в result: {len(result)}")

        if len(result) > 0:
            print(f"\n📋 Первый документ:")
            first_doc = result[0]
            for key, value in first_doc.items():
                print(f"   {key}: '{value}'")

            print(f"\n📋 Проверим еще несколько:")
            for i in range(1, min(4, len(result))):
                doc = result[i]
                url = doc.get('ID', 'НЕТ URL')
                desc = doc.get('Description', 'НЕТ ОПИСАНИЯ')
                parent = doc.get('Parent', 'НЕТ РОДИТЕЛЯ')
                date = doc.get('Date', 'НЕТ ДАТЫ')
                guid = doc.get('GuidDoc', 'НЕТ GUID')
                print(f"   Документ {i}: {desc[:50]}... | {parent} | {date} | URL: {url[:80]}...")
    else:
        print("❌ Нет поля 'result' в JSON")

except Exception as e:
    print(f"❌ Ошибка: {e}")