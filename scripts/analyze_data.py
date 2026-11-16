"""Generate a reproducible training-only EDA report (no model fitting)."""

import csv
import hashlib
import html
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from inspect_data import inspect


def grouped(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[int(row[key])].append(int(row["cnt"]))
    return [{"key": k, "rows": len(v), "mean": mean(v)} for k, v in sorted(groups.items())]


def table(headers, rows):
    def cell(value):
        return html.escape(str(value))
    return "<table><thead><tr>" + "".join(f"<th>{cell(h)}</th>" for h in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{cell(c)}</td>" for c in row) + "</tr>" for row in rows) + "</tbody></table>"


def main():
    raw = ROOT / "data" / "raw"
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    path = raw / "hour.csv"
    if inspect(path):
        raise SystemExit("Data inspection failed; report not generated.")
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    dates = sorted({r["dteday"] for r in rows})
    first, second = int(len(dates) * .70), int(len(dates) * .85)
    date_sets = {"train": set(dates[:first]), "validation": set(dates[first:second]), "test": set(dates[second:])}
    splits = {name: [r for r in rows if r["dteday"] in days] for name, days in date_sets.items()}
    assert set.union(*date_sets.values()) == set(dates)
    assert not (date_sets["train"] & date_sets["validation"] or date_sets["validation"] & date_sets["test"] or date_sets["train"] & date_sets["test"])
    assert max(date_sets["train"]) < min(date_sets["validation"]) < min(date_sets["test"])
    split_info = {name: {"start": min(days), "end": max(days), "days": len(days), "rows": len(splits[name])} for name, days in date_sets.items()}
    (reports / "split_manifest.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    train = splits["train"]
    hourly = grouped(train, "hr")
    weekdays = grouped(train, "weekday")
    weather = grouped(train, "weathersit")
    target = [int(r["cnt"]) for r in train]
    summary = {"scope": "Training data only; validation/test targets not explored", "splits": split_info, "training_target": {"mean": mean(target), "median": median(target), "min": min(target), "max": max(target)}, "hourly": hourly, "weekday": weekdays, "weather": weather}
    (reports / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    hashes = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in sorted(raw.iterdir()) if p.is_file() and p.name != ".gitkeep"}
    provenance = {"source": "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset", "download_url": "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip", "doi": "10.24432/C5W894", "license": "CC BY 4.0 per UCI dataset page", "citation": "Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine Learning Repository.", "paper": "Fanaee-T, H., Gama, J. Event labeling combining ensemble detectors and background knowledge (2013). doi:10.1007/s13748-013-0040-3", "manifest_generated_utc": datetime.now(timezone.utc).isoformat(), "files": hashes}
    (reports / "data_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for flag, label, color in [(1, "Working day", "#2563eb"), (0, "Weekend / holiday", "#d97706")]:
        values = grouped([r for r in train if int(r["workingday"]) == flag], "hr")
        ax.plot([v["key"] for v in values], [v["mean"] for v in values], marker="o", markersize=3, label=label, color=color)
    ax.set(xlabel="Hour of day", ylabel="Mean rentals per observed hour", title="Hourly demand profile — training data only", xticks=range(0, 24, 2))
    ax.grid(alpha=.18)
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports / "hourly_demand.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar([v["key"] for v in weekdays], [v["mean"] for v in weekdays], color="#2563eb")
    axes[0].set(xticks=range(7), xticklabels=["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"], title="Day of week", ylabel="Mean rentals per observed hour")
    bars = axes[1].bar([v["key"] for v in weather], [v["mean"] for v in weather], color="#0f766e")
    for bar, item in zip(bars, weather):
        axes[1].annotate(f'n={item["rows"]:,}', (bar.get_x()+bar.get_width()/2, bar.get_height()), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
    axes[1].set(xticks=[v["key"] for v in weather], xlabel="Original weather code (1–4)", title="Weather category", ylim=(0, max(v["mean"] for v in weather) * 1.25))
    fig.suptitle("Training-only associations; these do not establish causality")
    fig.tight_layout()
    fig.savefig(reports / "weekday_weather.png")
    plt.close(fig)
    peak = max(hourly, key=lambda x: x["mean"])
    sparse = min(weather, key=lambda x: x["rows"])
    split_table = table(["Часть", "Начало", "Конец", "Дней", "Строк"], [[name, v["start"], v["end"], v["days"], v["rows"]] for name, v in split_info.items()])
    weather_table = table(["Код погоды", "Часов в train", "Среднее аренд/час"], [[v["key"], v["rows"], f'{v["mean"]:.1f}'] for v in weather])
    document = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Bike Demand — данные</title>
<style>body{{font:17px/1.65 system-ui,sans-serif;color:#172b4d;background:#f3f6fa;margin:0}}main{{max-width:1050px;margin:auto;padding:36px 24px}}section{{background:white;padding:24px;margin:20px 0;border-radius:14px}}h1,h2{{line-height:1.2}}.tag{{color:#2563eb;font-weight:700}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dce3ec}}.note{{border-left:4px solid #d97706;padding-left:16px}}code{{overflow-wrap:anywhere}}</style>
<main><p class="tag">BIKE DEMAND · ЭТАП 2</p><h1>Что мы узнали из данных</h1><p>Почасовой спрос на Capital Bikeshare, Вашингтон, 2011–2012. Это анализ данных; модель ещё не обучалась.</p>
<section><h2>1. Проверка исходного файла</h2><p><b>{len(rows):,}</b> строк · <b>{len(dates)}</b> дней · <b>17</b> столбцов. Пустых значений, повторных временных отметок, отрицательных счётчиков и расхождений cnt = casual + registered не найдено.</p><p class="note">В полном временном диапазоне отсутствуют 165 часов. Причина не установлена. Не добавляем для них нулевой спрос. Метрики впоследствии будут относиться к наблюдаемым часам.</p></section>
<section><h2>2. Честное разделение по времени</h2>{split_table}<p>Границы определены по датам, без перемешивания. Все графики и статистика спроса ниже построены <b>только на train</b>. Target валидации и теста не исследуем. Сезонный состав частей различается — это ограничение одной временной проверки.</p></section>
<section><h2>3. Спрос зависит от времени</h2><img src="hourly_demand.png" alt="Средний спрос по часам отдельно для рабочих и нерабочих дней"><p>В среднем по train наиболее загружен час <b>{peak['key']:02d}:00</b>: <b>{peak['mean']:.1f}</b> аренд на наблюдаемый час. Рабочие и нерабочие дни имеют разные профили. Поэтому модель должна учитывать сочетание часа и типа дня.</p><p>Среднее по train: <b>{mean(target):.1f}</b>, медиана: <b>{median(target):.0f}</b>, диапазон: <b>{min(target)}–{max(target)}</b> аренд/час. Одно постоянное число не описывает колебания спроса.</p></section>
<section><h2>4. День недели и погода</h2><img src="weekday_weather.png" alt="Средний спрос по дням недели и категориям погоды">{weather_table}<p>Коды: 1 — ясно / небольшая облачность; 2 — туман / облачность; 3 — небольшой дождь / снег; 4 — тяжёлые погодные условия, согласно исходному описанию.</p><p class="note">Для кода {sparse['key']} в train всего {sparse['rows']} наблюдений. По такой выборке нельзя делать устойчивые выводы. Средние по погоде также зависят от часа и сезона; они не доказывают причинное влияние.</p></section>
<section><h2>5. Решения перед обучением</h2><ul><li>Ответ — cnt. Исключаем casual и registered: их сумма уже равна ответу. Исключаем instant как порядковый номер.</li><li>Базовое сравнение — средний спрос для часа недели, рассчитанный на train; при отсутствии группы — общее среднее train.</li><li>Сравним Ridge и градиентный бустинг. Подбор — по validation MAE; финальный test оставляем на конец.</li><li>Погода в исходных данных фактически наблюдалась. Сервис будет оценкой сценария с заданной погодой, а не проверенным прогнозом на завтра.</li></ul></section>
<section><h2>6. Ограничения и происхождение</h2><p>Описание на текущей странице UCI и Readme.txt архива расходится: число строк на странице — 17 389, фактически — 17 379; также различаются описания season и нормализации temp/atemp. Пока сохраняем исходные коды и нормализованные величины, не переводим их в °C и не присваиваем сезонам подписи.</p><p>Данные исторические и агрегированы по системе. Прогноз доступности конкретной станции и применимость к современному городу здесь не проверяются.</p><p><a href="https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset">Источник: UCI Bike Sharing</a> · DOI: 10.24432/C5W894 · CC BY 4.0 по странице UCI. Автор: Hadi Fanaee-T (2013). Статья: Fanaee-T, H., Gama, J., DOI: 10.1007/s13748-013-0040-3.</p><p>Контрольные суммы файлов: <code>reports/data_manifest.json</code>. Числовые результаты: <code>reports/eda_summary.json</code>. Повторить: <code>.venv\\Scripts\\python.exe scripts\\analyze_data.py</code>.</p></section></main></html>'''
    (reports / "data_report.html").write_text(document, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Report: {reports / 'data_report.html'}")


if __name__ == "__main__":
    main()
