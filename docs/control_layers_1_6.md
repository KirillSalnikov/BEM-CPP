# Контрольные слои 1-6

Цель этих файлов - убрать ручную подгонку и сделать проверку точности/знаков
воспроизводимой. Быстрый CUDA/FMM режим не должен считаться правильным сам по
себе: он проходит через независимые малые проверки, контроль сетки и сравнение
с плотным решением.

## 1. CPU reference PMCHWT

Файл: `scripts/cpu_pmchwt_centroid_reference.py`.

Это независимый диагностический CPU reference для очень малых сеток. Он строит
RWG-базис, собирает приближенные операторы `L` и `K` на centroid-квадратуре и
из них формирует несколько систем: `pmchwt`, `balanced`, `muller`,
`muller-balanced`, `muller2-balanced`. Для каждой системы пишутся масштабы
неизвестных, знак внутреннего оператора, identity jump, N-form флаг и оценка
обусловленности. Сингулярные self-термы там намеренно пропущены, поэтому это
не production-accuracy solver. Его задача - ловить грубые ошибки знаков,
индексации блоков, несогласованную балансировку и деградацию обусловленности
без CUDA.

Команда:

```bash
python3 scripts/cpu_pmchwt_centroid_reference.py --ka 1 --ri 1.3116 0
python3 scripts/cpu_pmchwt_centroid_reference.py --system muller2-balanced --ka 1 --ri 1.3116 0
python3 tests/test_cpu_pmchwt_centroid_reference.py
```

## 2. Mueller/sign audit

Файлы: `scripts/mueller_audit.py`, `tests/test_mie_mueller_symmetry.py`,
`tests/test_mueller_audit_physical.py`.

Проверяет форму массива Mueller, конечность чисел, отрицательную `M11`,
малые значения `M11`, превышение `|Mij| / M11`, пассивные ограничения
поляризуемости и дихроизма, а также относительные L2-ошибки против ADDA или
MBS, если передан эталон. Диагностически также считается минимальное
собственное значение матрицы когерентности Cloude; флаг
`--require-cloude-physical` делает этот критерий строгим.

Команды:

```bash
python3 scripts/mueller_audit.py --self-test
python3 tests/test_mie_mueller_symmetry.py
python3 tests/test_mueller_audit_physical.py
python3 scripts/mueller_audit.py --bem result.json --adda /path/to/adda/raw --beta-order 0 --max-l2 0.1
```

`tests/test_mie_mueller_symmetry.py` является аналитическим sphere/Mie
регрессионным тестом: на сетке 0--180 градусов для слабопоглощающей сферы он
проверяет `M12=M21`, `M34=-M43`, `M33=M44`, нулевые off-block элементы,
`|Mij| <= M11` и инвариант амплитудного представления
`M11^2-M12^2=M33^2+M34^2`. Это ловит перестановки знаков и индексов, которые
могут не проявиться в одной только фазовой функции.

`tests/test_mueller_audit_physical.py` проверяет, что заведомо физическая
матрица проходит, а матрицы с поляризуемостью, дихроизмом или отдельным
элементом больше `M11` отклоняются. Это защищает быстрые режимы усреднения и
постобработки от формально конечных, но физически невозможных результатов.

Для production-очереди отдельный быстрый gate встроен в
`scripts/check_result_metadata.py --strict --require-converged --validate-numeric --require-cloude-physical`.
Он не заменяет сравнение с ADDA/Mie, но не дает пометить результат как
`current`, если JSON неполный: `theta` пустой, `mueller` не имеет 16 элементов
на угол, содержит `NaN/Inf`, длина массивов не совпадает или `M11` стал
отрицательным за пределом малого численного допуска. Дополнительно проверяются
физические bound `|M_{ij}| <= M_{11}`,
`sqrt(M21^2+M31^2+M41^2)/M11 <= 1` и
`sqrt(M12^2+M13^2+M14^2)/M11 <= 1` с малым численным запасом; это ловит ошибки
знаков, нормировки или перестановки компонент, когда `M11` выглядит
правдоподобно, а остальные элементы уже невозможны. В guarded production
runner и queue-status проверка Cloude включена по умолчанию через
`--require-cloude-physical`: дополнительно проверяется положительная
полуопределенность матрицы когерентности Cloude для каждого угла. Временный
обход для legacy JSON должен быть явным: `BEM_METADATA_SKIP_CLOUDE=1`.

## 3. Singular / near-singular integrals

Файл: `scripts/near_singular_audit.py`.

Это еще не полный Duffy-интегратор, но теперь это не только внешний скрипт:
та же классификация попадает в `MeshQualityReport`, `--mesh-quality-report` и
основной JSON результата. Скрипт и C++ mesh-gate разделяют пары на:

- `self_panel_count`: сингулярные self-интегралы;
- `edge_adjacent_pair_count`: пары с общим ребром;
- `vertex_adjacent_pair_count`: пары с общей вершиной;
- `near_disjoint_pair_count`: близкие, но топологически несмежные панели;
- `far_disjoint`: обычная квадратура.

Это важнее простого расстояния центроидов: self/edge/vertex пары требуют
сингулярной или Duffy-квадратуры даже тогда, когда центроиды не попали под
порог близости. Поле `taylor_duffy_candidate_count` дает нижнюю оценку числа
пар, которые нельзя честно вести через общий `quad4/7/13` путь.

Новые обязательные поля в `mesh`:

```text
self_panel_count
edge_adjacent_pair_count
vertex_adjacent_pair_count
near_disjoint_pair_count
taylor_duffy_candidate_count
recommended_min_quad_order
recommended_mesh_strategy
recommended_mesh_action
requires_remesh
```

`taylor_duffy_candidate_count` равно сумме self, edge-adjacent,
vertex-adjacent и near-disjoint классов. `scripts/check_result_metadata.py`
проверяет эту сумму и не принимает `current` результат, если
`near_disjoint_pair_count > 0`: такой расчет требует либо исправления сетки,
либо отдельного near-singular пути, а не обычного `quad4/7/13`.
Поле `recommended_min_quad_order` также проверяется против `method.quad_order`:
например, частица с sharp-ребрами получает стратегию
`edge_aware_refinement` и минимум `quad7`, а гладкая сфера может остаться на
`uniform_curvature_refinement` и `quad4`. Если `requires_remesh=true`, результат
не считается принятым.

Команды:

```bash
python3 scripts/near_singular_audit.py mesh.obj --threshold 0.75
python3 tests/test_near_singular_audit.py
python3 scripts/check_result_metadata.py result.json --strict --require-converged --validate-numeric --require-cloude-physical
```

## 4. Mesh quality gate

Файлы: `src/mesh.h`, `src/mesh.cpp`, `src/main.cpp`.

Новые флаги:

```bash
--mesh-quality-report FILE
--mesh-quality-strict
--mesh-quality-only
```

Gate проверяет замкнутость, ориентацию нормалей, вырожденные треугольники,
минимальные углы, aspect ratio, статистику ребер и площадей. Дополнительно
пишется edge-aware диагностика: число feature-ребер с диэдральным углом
`>=30°`, максимальный диэдральный угол, средний угол sharp-ребер и максимальный
скачок площади двух треугольников, соседних по ребру. Эти поля есть и в
`--mesh-quality-report`, и в основном JSON результата:

```text
feature_edges_30deg
max_dihedral_deg
mean_feature_dihedral_deg
max_adjacent_area_ratio
```

Они нужны для автоматического выбора mesh strategy: сфера с хорошим refinement
не должна иметь feature-ребер, а призма или пылевая частица обязаны явно
показывать sharp zones, где refinement должен контролировать не только размер,
но и качество соседних треугольников. В строгом режиме расчет не стартует на
плохой сетке.

Дополнительно C++ mesh-gate пишет машинную рекомендацию:

```text
recommended_mesh_strategy
recommended_mesh_action
recommended_min_quad_order
requires_remesh
```

Правило простое и воспроизводимое: гладкие поверхности идут через
`uniform_curvature_refinement`, sharp-edge формы через `edge_aware_refinement`
с минимумом `quad7`, а near-touch/плохая топология/skinny triangles требуют
remesh до production-расчета.

## 5. Operator architecture

Файл: `src/operator_config.h`.

Формирование PMCHWT-блоков и выбор масштабов системы вынесены из `main.cpp` и
`src/pmchwt.cu` в один контракт. `bem_block_scales_for_system` задает
`unknown_m_scale`, `row_h_scale`, знак внутреннего оператора, identity jump и
N-form параметры для `pmchwt`, `balanced`, `muller`, `muller-balanced` и
экспериментального `muller2-balanced`. После этого dense, FMM и CPU-аудит
должны сравнивать один и тот же блок, а preconditioner выбирается отдельной
измеряемой политикой и не может менять физическую систему:

```text
[ eta_e L_e + eta_i L_i      -(K_e + K_i) ]
[ K_e + K_i                  L_e/eta_e + L_i/eta_i ]
```

Для обобщенных систем тот же блок записывается через масштабы:

```text
K_s = K_e + s_i K_i + c_I I

[ eta_e L_e + s_i eta_i L_i           -K_s / s_M ]
[ r_H K_s             r_H (L_e/eta_e + s_i L_i/eta_i) / s_M ]
```

где `s_i` - знак внутреннего оператора, `c_I` - identity jump,
`s_M=unknown_m_scale`, `r_H=row_h_scale`. Файл
`scripts/operator_block_audit.py` проверяет алгебру блоков на синтетических
матрицах. `tests/operator_config_check.cpp` прогоняет C++-контракт сборки
блоков для `pmchwt`, `balanced`, `muller`, `muller-balanced` и
`muller2-balanced`, а `tests/test_cpu_pmchwt_centroid_reference.py` проверяет
тот же набор систем на независимой маленькой RWG-сборке. Вместе эти проверки
гарантируют, что выбранный `--system` меняет фактическую матрицу, а не только
подпись в результате.

Для прекондиционера metadata checker теперь проверяет не только
`preconditioner_enabled`, но и `schwarz_preconditioner` плюс согласованную
`preconditioner_reason`. Включенный прекондиционер допускает только причины
`auto` или `forced`; выключенный должен иметь одну из причин политики
`choose_precond_policy`, например `small_nonsphere`, `hex_unpreconditioned_faster`
или `obj_ka_ge_4_unpreconditioned_measured`. Если `preconditioner_enabled=false`,
то `schwarz_preconditioner` также обязан быть `false`; иначе результат
считается противоречивым. Это не дает принять результат, где прекондиционер был
включен или отключен не той веткой политики.

## 6. Python job API

Файл: `bemcuda/job.py`.

`BemJob` строит явную команду запуска без скрытых параметров. Это нужно для
очередей и постерных расчетов: параметры расчета сериализуются и могут быть
повторены без копирования длинных shell-команд. API поддерживает старый плоский
стиль аргументов, но для новых расчетов нужно использовать структурированный
слой:

- `Material`: комплексный показатель преломления;
- `Geometry`: форма, `ref`, OBJ-сетка, разбиение, параметры призмы;
- `OrientationGrid`: одиночный расчет, сетка Эйлера, explicit файл, chunk
  `orient-start/count`, `alpha-avg`;
- `SolverOptions`: backend, система, квадратура, `fmm-digits`, GMRES tolerance,
  restart, leaf size, preconditioner policy;
- `MeshQuality`: строгий mesh-gate и JSON-отчет.

`BemJob.manifest()` записывает не только параметры, но и итоговую команду,
окружение GPU (`BEM_ASM_GPU_LIST`, `BEM_LU_GPU_LIST`, `BEM_FF_GPU_LIST`) и
`semantic_id`. Этот идентификатор не зависит от пути выходного JSON и binary, но
меняется при изменении физики или численных параметров. Его надо хранить в
очередях и таблицах постера, чтобы не смешивать результаты с разными `quad`,
`gmres_tol`, системой или ориентационной сеткой.

Минимальный пример:

```python
from pathlib import Path
from bemcuda import (
    BemJob,
    Geometry,
    Material,
    MeshQuality,
    OrientationGrid,
    SolverOptions,
)

job = BemJob(
    ka=5,
    material=Material((1.3116, 0.0)),
    geometry=Geometry(shape="hex_prism", ref=2, prism_aspect=1.5),
    orientations=OrientationGrid(single=True),
    solver_options=SolverOptions(backend="fmm", system="balanced", quad=7),
    out=Path("runs/hex5.json"),
    mesh_quality=MeshQuality(strict=True, report=Path("runs/hex5_mesh.json")),
)
print(" ".join(job.command()))
print(job.semantic_id())
```

Проверка API:

```bash
python3 tests/test_bem_job_api.py
```

Она проверяет генерацию команд для одиночной ориентации, ориентационного файла,
OBJ-сетки, `--mesh-quality-*`, `--no-prec` и JSON-сериализацию параметров.

## Общие проверки

Локально без CUDA:

```bash
make audit-1-6
```

Цель `audit-1-6` запускает `host-audits`, затем записывает сводный JSON в
`runs/audit_1_6_report.json` и проверяет его структуру через
`scripts/check_audit_1_6_report.py`. Для отладки те же шаги можно запускать
отдельно:

```bash
scripts/run_local_audits.sh
python3 scripts/audit_1_6.py --out runs/audit_1_6_report.json
python3 scripts/check_audit_1_6_report.py runs/audit_1_6_report.json
```

Короткая расшифровка отчета:

```bash
make audit-1-6-summary
```

На сервере с CUDA:

```bash
make cuda-runtime-check
make cuda-audits
make cuda-audits-summary
```

Цель `cuda-runtime-check` быстро проверяет toolkit, `/dev/nvidia*`,
`libcuda.so` и драйвер без запуска dense-vs-FMM. JSON диагностика пишется в
`runs/audit_1_6_cuda/cuda_runtime_detect.json`.
Цель `cuda-audits` вызывает `scripts/run_cuda_reference_audits.sh`.
Если `bin/bem_cuda_fmm` отсутствует, runner сам запускает
`scripts/build_cuda_fmm.sh`. Сборщик находит `CUDA_HOME` через
`scripts/detect_cuda_toolchain.py --print-env` и выбирает совместимый host
compiler для `nvcc`. Это важно для CUDA 12.2: на сервере может стоять более
новый системный GCC, а `nvcc` должен компилировать через GCC/G++ 12. После
создания `runs/audit_1_6_cuda/report.json` runner проверяет его структуру через
`scripts/check_audit_1_6_report.py`.

`scripts/audit_1_6.py` пишет JSON-отчет с состоянием каждого пункта. В отчете
отдельно указаны:

```text
cuda_toolchain_available - найден nvcc, заголовки и libcudart
cuda_runtime_ready       - доступны /dev/nvidia*, libcuda.so и рабочий драйвер
```

Если CUDA toolkit найден, но драйвер/GPU недоступны на текущей машине,
dense-vs-FMM проверка помечается как пропущенная с причиной, а не как успешная.
Полный runner после этого делает строгую проверку runtime и завершается с кодом
`3`, чтобы такой запуск нельзя было принять за проверенный CUDA reference.
Если нужен именно этот код выхода, запускайте runner напрямую:
`scripts/run_cuda_reference_audits.sh`. Через `make cuda-audits` GNU make
покажет ошибку цели, но собственный код выхода `make` может быть общим для
любых ошибок сборки.
Цель `cuda-audits-summary` читает `runs/audit_1_6_cuda/report.json` в строгом
режиме: если dense-vs-FMM reference не подтвержден, она возвращает ошибку.

На GPU-сервере строгий режим аудита должен требовать CUDA reference явно:

```bash
python3 scripts/audit_1_6.py --run-cuda --require-cuda-reference --binary ./bin/bem_cuda_fmm
```

Если локальные пункты 1-6 прошли, но dense-vs-FMM reference не проверен, этот
режим возвращает код `4`.

Если CUDA установлена не в `/usr/local/cuda`, сначала можно проверить toolchain:

```bash
python3 scripts/detect_cuda_toolchain.py
python3 scripts/detect_cuda_toolchain.py --print-env
```

Runner `scripts/run_cuda_reference_audits.sh` сам вызывает этот детектор, если
`CUDA_HOME` не задан. Детектор поддерживает как обычные CUDA roots
`/usr/local/cuda` и conda-env, так и системную раскладку Debian/WSL, где
`nvcc` находится в `/usr/bin`, заголовки в `/usr/include`, а `libcudart.so` в
`/usr/lib/x86_64-linux-gnu`.

Строгая проверка runtime:

```bash
python3 scripts/detect_cuda_toolchain.py --require-runtime
```

Коды выхода:

```text
0 - toolkit и runtime готовы
2 - toolkit не найден или неполный
3 - toolkit найден, но NVIDIA runtime/driver не готов
```

Для сборки плотных exterior/interior операторов можно явно выбрать карты:

```bash
BEM_ASM_GPU_LIST=4 ./bin/bem_cuda_fmm ...
BEM_ASM_GPU_LIST=2,4 ./bin/bem_cuda_fmm ...
```

Один номер закрепляет всю сборку за одной картой. Два номера считают внешний и
внутренний операторы параллельно на двух GPU. Это полезно, когда одна карта
перегревается, отваливается или занята другим расчетом. Старый флаг
`BEM_ASM_MGPU=1/2` оставлен как совместимость, но для воспроизводимых запусков
лучше задавать именно `BEM_ASM_GPU_LIST`.

Для production-матрицы точности деплой и сбор результатов сделаны
неинтерактивными. Если сервер сменил адрес, можно задать его явно и отключить
старт очереди:

```bash
REMOTE_HOST=172.16.1.222 START_QUEUE=0 START_POWER_WATCH=0 RUN_PREFLIGHT=0 \
  scripts/deploy_accuracy_matrix_15_queue.sh
REMOTE_HOST=172.16.1.222 scripts/fetch_accuracy_matrix_15_results.sh
```

`fetch_accuracy_matrix_15_results.sh` после копирования всегда запускает
локальный `scripts/audit_accuracy_matrix_15.py` и печатает `FETCH_AUDIT_RC`.
По умолчанию это строгий режим: если точность, метаданные или provenance
оператора еще не проходят контроль, fetch возвращает тот же ненулевой код. Для
диагностики можно забрать файлы без аварийного завершения:

```bash
REMOTE_HOST=172.16.1.222 scripts/fetch_accuracy_matrix_15_results.sh --audit-best-effort
```

Когда доступно несколько GPU-хостов, новые case лучше раскладывать снаружи,
а не запускать один расчет сразу на нескольких картах. Скрипт ниже проверяет
каждую указанную карту через SSH, отбрасывает занятые/горячие GPU и назначает
разные case на разные `host:gpu`:

```bash
scripts/remote_resume_accuracy_matrix_cases.sh \
  --hosts "gpu1 gpu2 epyc1 172.16.1.222" \
  --gpus "0" \
  --cases "hex_ka30_ref5_balanced_q7_d5_tol1e3,sphere_ka30_ref6_current_q7_d6_tol3e3" \
  --max-jobs 2
```

По умолчанию это dry-run. Для реального старта добавляется `--run`. В одном
вызове один `host:gpu` получает не больше одного расчета, а удаленная команда
явно запускается с `BEM_NO_AUTO_MGPU=1`, `--gpus <одна карта>` и
`--max-jobs 1`. Поэтому масштабирование production-очереди идет по схеме
разные case на разные GPU, а не один case на несколько GPU. Это защищает от
ситуации, когда две тяжелые задачи одновременно занимают одну карту или один
расчет расползается по нескольким картам и валит сервер по питанию или
температуре. Пороги занятости задаются через
`--max-temp`, `--max-util`, `--max-mem`, а защитные лимиты самого case-runner
передаются через `--case-max-power`, `--case-max-temp`,
`--case-max-bad-samples`. Хост выбирается только если на нем найден рабочий
checkout с `scripts/resume_accuracy_matrix_cases.sh` и исполняемый
`bin/bem_cuda_fmm.next` или `bin/bem_cuda_fmm`; старые checkout без production
runner или без бинарника не получают задания.

По умолчанию deploy делает preflight, записывает план очереди в
`runs/production_matrix_15/queue.plan` и не стартует вторую очередь, если
`runs/production_matrix_15/queue.pid` указывает на живой процесс.

Для живой диагностики очереди:

```bash
scripts/queue_live_status.sh
scripts/queue_watch_once.sh
python3 scripts/queue_status_json.py --out runs/production_matrix_15 > runs/production_matrix_15/status.json
```

Human-readable статус показывает `queue.pid`, дочерний solver-процесс, текущий список
`CURRENT/MISSING`, загрузку GPU, возраст логов и последние строки
`*.gpu.csv`. Это помогает отличить долгий GMRES solve от зависшего процесса,
даже если сам solver не печатает итерации в лог. Для новых verbose-запусков
статус дополнительно выводит последнюю найденную строку GMRES вида
`gmres_last=GMRES iter ...` или финальный `gmres_done=DONE ...`.
`queue_watch_once.sh` пишет `runs/production_matrix_15/status.json`, печатает
короткую сводку и возвращает строгий код только для реальных проблем живой
очереди: `20` для `FAIL` в логе solver, `21` для остановившегося monitor CSV и
`25`, если `queue.pid` уже не живой, но в плане еще есть `missing` или `stale`
результаты.
JSON-вариант содержит те же counts, список результатов, monitor summaries,
`active_monitors`, `failed_monitors` и `stalled_monitors`; его можно читать из
plotting/audit скриптов без парсинга human-readable вывода. При повторных
запусках он также пишет `delta.wall_s` и `sample_delta` по каждому monitor CSV,
чтобы машинно отличать живой прогресс от остановившегося процесса.

Для автоматической вахты у JSON-статуса есть строгие exit-коды:

```bash
scripts/queue_watch_once.sh
```

Коды: `20` для `FAIL` в логе solver, `21` для активного monitor CSV без
прироста после `--stall-wall-s`, `22` для отсутствующих результатов,
`23` для stale/metadata-invalid результатов, `24` для любого незаконченного
пункта (`missing` или `stale`), `25` для остановленной очереди при незакрытом
плане. Для живой очереди `queue_watch_once.sh` по умолчанию включает `20`, `21`
и `25`; `22`/`23`/`24` нужны только для финального контроля после завершения.

В production queue новые запуски по умолчанию включают
`BEM_GMRES_VERBOSE=1`, поэтому в `runs/production_matrix_15/logs/*.log`
появляются остатки GMRES по итерациям/рестартам. Если нужен компактный лог,
это можно отключить только для очереди:

```bash
BEM_QUEUE_GMRES_VERBOSE=0 scripts/run_accuracy_matrix_15_queue.sh --run-missing
```

Чтобы эти строки появлялись в файле сразу, а не после завершения процесса,
queue-runner по умолчанию запускает solver через `stdbuf -oL -eL`. Это можно
отключить для диагностики:

```bash
BEM_QUEUE_STDBUF=0 scripts/run_accuracy_matrix_15_queue.sh --run-missing
```

GPU-монитор не должен менять семантику завершения расчета. Если solver падает,
`run_with_gpu_monitor` возвращает исходный код ошибки solver-а, а не гасит его
внутри `wait` под `set -e`. Это нужно, чтобы failed case был виден очереди как
реальный отказ, а не как тихий обрыв мониторинга. В логе такого case
дополнительно пишется строка `FAIL имя_case rc=код`.

Если старая очередь уже запущена, не надо подменять файл
`scripts/run_accuracy_matrix_15_queue.sh` под живым процессом. Для этого есть
wrapper, который сначала проверяет удаленное состояние, а затем может дождаться
завершения текущей очереди и запустить deploy заново:

```bash
REMOTE_HOST=172.16.1.222 scripts/resume_accuracy_matrix_15_after_current.sh --status-only
REMOTE_HOST=172.16.1.222 WAIT_INTERVAL_S=60 \
  scripts/resume_accuracy_matrix_15_after_current.sh --wait-and-resume
REMOTE_HOST=172.16.1.222 WAIT_INTERVAL_S=60 \
  scripts/resume_accuracy_matrix_15_after_current.sh --install-remote-watcher
```

Это важно после сбоев питания или старых запусков с агрессивной остановкой по
показаниям GPU. Новый queue-runner только пишет мониторинг в
`runs/production_matrix_15/logs/*.gpu.csv` и не останавливает расчет по
температуре, памяти или мощности. Решение о снижении нагрузки принимается
вручную по этому логу и по данным BMC.

Для аварийного продолжения только выбранных точек матрицы используется
case-runner: один case назначается на одну GPU, без разбиения одного расчета
между несколькими картами. `resume_accuracy_matrix_cases.sh` также выставляет
`BEM_NO_AUTO_MGPU=1` перед запуском `run_accuracy_matrix_case.sh`, так что
каждый дочерний solver видит только назначенную ему карту. Например, чтобы
после перезапуска досчитать только две тяжелые точки, а не начинать всю матрицу
сначала:

```bash
cd /home/kirill_epyc/BEM-CUDA
scripts/resume_accuracy_matrix_cases.sh --run \
  --gpus "0 1" \
  --cases "hex_ka30_ref5_balanced_q7_d5_tol1e3,sphere_ka30_ref6_current_q7_d6_tol3e3" \
  --case-max-power 290 \
  --case-max-bad-samples 4
```

Если результат есть, но не проходит текущие проверки метаданных, complex-operator
или GMRES-сходимости, resume-runner считает его `STALE` и запускает case с
`--force`, архивируя старый JSON и лог. По умолчанию oversubscribe запрещен:
один вызов стартует не больше задач, чем выбранных свободных GPU.

После аудита точности следующий уровень уточнения строится отдельным
планировщиком:

```bash
python3 scripts/plan_accuracy_refinement_cases.py \
  --csv poster_a0/assets/table_accuracy_matrix_15.csv \
  --gpus "0 1 2 3"
```

Он выбирает только строки, которые не прошли заданный порог или имеют
устаревшую provenance-метаинформацию, и печатает параметрические case-имена
следующего уровня, например `sphere_ka30_ref7_current_q9_d7_tol1e3`. Эти имена
принимаются `scripts/run_accuracy_matrix_case.sh` и
`scripts/resume_accuracy_matrix_cases.sh`. Если точность уже проходит, но
метаданные старые, планировщик повторяет тот же численный уровень. Статус
`metadata=ok` в accuracy audit строится тем же строгим gate, что и очередь:
`--strict --require-converged --validate-numeric --require-cloude-physical`
плюс contract case-имени; для пылевых absorbing case дополнительно требуется
complex-operator provenance. Это требование включено по умолчанию; старые
absorbing результаты без `row_h_scale_complex` допускаются только явным
legacy-флагом `--allow-missing-complex-operator-for-absorbing`. Если ошибка
выше порога, он повышает `ref`, `quad`, `digits` и ужесточает
`gmres_tol` без ручного редактирования shell-скриптов. По умолчанию он
печатает не больше задач, чем GPU указано в `--gpus`; полный список
доступен через `--all-cases`. В сгенерированную команду для
`resume_accuracy_matrix_cases.sh` также добавляется соответствующий
`--max-jobs`, чтобы запускатель не стартовал больше задач за один проход.
CSV-план сохраняет причину пересчета (`accuracy`, `metadata` или обе) и
исходный файл результата, из которого принято решение. По умолчанию он
пишется в `runs/production_matrix_refinement/plan.csv`; путь можно заменить
через `--plan-csv` или отключить запись через `--no-plan-csv`.
Физическая постановка при этом не меняется.

Снимок состояния питания и BMC без бесконечного цикла:

```bash
scripts/remote_power_watch.sh --once
```

В постоянном режиме `remote_power_watch.sh 10` проверяет passwordless sudo один
раз за цикл. Если sudo без пароля не настроен, он пишет одну диагностическую
строку в каждый IPMI-раздел, но не запускает три отдельные sudo-проверки подряд
и не засоряет journal повторными auth-failure записями.

Сводка по GPU-мониторингу:

```bash
python3 scripts/summarize_gpu_power_monitor.py runs/production_matrix_15/logs
python3 scripts/summarize_gpu_power_monitor.py --json runs/production_matrix_15/logs
```

Она выводит число сэмплов, длительность, среднюю, p95 и максимальную мощность,
максимальную температуру и память для каждого `*.gpu.csv`.

Ограничение мощности GPU задается отдельно от мониторинга:

```bash
REMOTE_HOST=172.16.1.222 scripts/set_remote_gpu_power_limit.sh --show
REMOTE_HOST=172.16.1.222 GPU_POWER_LIMIT_W=200 \
  scripts/set_remote_gpu_power_limit.sh --set
```

`--set` вызывает `sudo -n nvidia-smi -pl`; если passwordless sudo не настроен,
скрипт не меняет лимиты и печатает команды, которые надо выполнить на сервере
после ввода пароля.

Queue-runner выбирает бинарник так: если `BIN` явно не задан и существует
исполняемый `./bin/bem_cuda_fmm.next`, используется он; иначе используется
`./bin/bem_cuda_fmm`. Это позволяет подготовить и проверить новый бинарник рядом
с работающей очередью, не перезаписывая файл, который уже мог быть использован
живым процессом.

Диагностика BMC Supermicro и вход в админку вынесены в отдельный скрипт:

```bash
scripts/supermicro_bmc_access.sh --print
REMOTE_HOST=172.16.1.222 scripts/supermicro_bmc_access.sh --remote-diagnose
```

Если BMC `192.168.0.103` не пингуется с сервера, браузер и SSH-туннель тоже не
помогут. Тогда сначала на сервере надо добавить адрес в подсеть BMC:

```bash
sudo ip addr add 192.168.0.10/24 dev eth0 2>/dev/null || true
sudo ip route replace 192.168.0.0/24 dev eth0 src 192.168.0.10
ping -c 3 192.168.0.103
```

После этого с рабочей машины:

```bash
ssh -N -L 8443:192.168.0.103:443 kirill_epyc@172.16.1.222
```

и открыть `https://localhost:8443`. Записи BMC вида `ACPowerOn(OEM) First AC
Power on` означают, что BMC заново увидел подачу сетевого AC-питания. Это
сильный признак внешнего отключения/просадки питания, PDU/розетки/БП или
защиты, а не обычного `shutdown`, OOM или ошибки CUDA в ядре.

Удаленное включение сервера делается не через SSH на сервер, а напрямую через
BMC/IPMI. Если ОС выключена, SSH-туннель через этот же сервер недоступен, но
BMC должен отвечать отдельно:

```bash
ping 192.168.0.103
scripts/ipmi_power_control.sh --check
scripts/ipmi_power_control.sh --print
```

`--check` проверяет профиль BMC, маршрут до BMC, ping, наличие локального
`ipmitool` и файл пароля. Если локального `ipmitool` нет, полноценное включение
после выключения ОС еще не настроено: временный запуск IPMI через сам сервер
работает только пока этот сервер уже включен.

Профиль BMC создается один раз:

```bash
BMC_HOST=192.168.0.103 BMC_USER=ADMIN scripts/ipmi_power_control.sh --setup-profile
scripts/ipmi_power_control.sh --check
```

После этого проверка питания и включение не требуют повторять адрес и файл
пароля:

```bash
scripts/ipmi_power_control.sh --status
scripts/ipmi_power_control.sh --on
scripts/ipmi_power_control.sh --wait-on
```

Эквивалентные команды `ipmitool`:

```bash
ipmitool -I lanplus -H 192.168.0.103 -U ADMIN -f ~/.config/bemcuda/bmc.pass chassis power status
ipmitool -I lanplus -H 192.168.0.103 -U ADMIN -f ~/.config/bemcuda/bmc.pass chassis power on
```

Если на локальной машине `ipmitool` еще не установлен, а сервер сейчас включен,
можно временно выполнить IPMI-команду через сервер, где `ipmitool` уже есть:

```bash
BMC_VIA_SSH_HOST=172.16.1.222 \
  BMC_PASS_FILE=~/.config/bemcuda/bmc.pass \
  scripts/ipmi_power_control.sh --status
```

Этот режим годится для проверки BMC и `power status`, но не заменяет настоящее
удаленное включение после полного выключения именно этого сервера: если ОС на
нем не запущена, `BMC_VIA_SSH_HOST=172.16.1.222` тоже недоступен.

Для плотного solve и ориентационного усреднения есть такие же явные списки:

```bash
BEM_LU_GPU_LIST=2,4 BEM_FF_GPU_LIST=2,4 ./bin/bem_cuda_fmm ...
```

`BEM_LU_GPU_LIST` делит несколько правых частей LU-solve между указанными GPU.
`BEM_FF_GPU_LIST` делит пакетное накопление матрицы Мюллера при
`alpha-avg`/геометрическом усреднении. Если список не задан, остается старый
авто-режим через `BEM_LU_MGPU`, `BEM_FF_MGPU` и `BEM_NO_AUTO_MGPU`.

Если локально есть только часть CUDA-пакетов, не надо чинить это вручную
символическими ссылками. Создать отдельную среду:

```bash
conda env create -f environment.cuda.yml
conda activate bem-cuda-toolchain
make cuda-audits
```

`environment.cuda.yml` фиксирует CUDA 12.2 и GCC/G++ 12. Такая среда нужна
именно для сборки. Для фактического CUDA reference все равно требуется машина с
NVIDIA-драйвером, `/dev/nvidia*` и `libcuda.so`; без этого локальный аудит
останется валидным, но dense-vs-FMM сравнение будет только отложено.
