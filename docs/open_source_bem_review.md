# Обзор открытых BEM/PMCHWT реализаций и сравнение с BEM-CUDA

Дата: 2026-06-23

## Что проверено

- SCUFF-EM: локально просмотрен shallow clone `HomerReid/scuff-em`, README, `libs/libscuff`, `libs/libhmat`, тесты Mie/Fresnel.
- Bempp-cl: локально просмотрен shallow clone `bempp/bempp-cl`, README, `bempp_cl/api`, `bempp_cl/core`, `api/fmm`, OpenCL/Numba kernels.
- MNPBEM: локально просмотрен shallow clone `Nikolaos-Matthaiakakis/MNPBEM`, README, Matlab-классы `BEM`, `Greenfun`, `Mesh2d`, `Mie`, MEX/H-matrix части.
- nanobem22: локально просмотрен shallow clone `uhohenester/nanobem22`, README, `Boundary`, `Material`, `particleshapes`, `Misc/integration`, help по stat/ret BEM.
- PMCHWT toy implementation: локально просмотрен shallow clone `sekulicivan7/PMCHWT`, `PMCHWT.cpp`, `EFIEop.cpp`, `MFIEop.cpp`.
- HELIOS: локально просмотрен shallow clone `mavrikak/Helios`, README, `src/SIEFormPMCHW.cpp`, `src/SimJob.cpp`, `src/RWGFun.cpp`, `src/SingSub.cpp`, `src/SingCancellation.cpp`, `include/Domain*.h`, `include/GreenF*.h`, Python tools/GUI.
- `scattering_random_orientation`: репозиторий, упоминавшийся для ориентационного усреднения поверх SCUFF-EM, не клонируется без авторизации; в сравнение не включен как проверенный код.

Основные внешние ссылки: SCUFF-EM <https://github.com/HomerReid/scuff-em>, HELIOS <https://github.com/mavrikak/Helios>, Bempp-cl <https://github.com/bempp/bempp-cl>, MNPBEM <https://github.com/Nikolaos-Matthaiakakis/MNPBEM>, nanobem22 <https://github.com/uhohenester/nanobem22>, PMCHWT <https://github.com/sekulicivan7/PMCHWT>, HELIOS preprint <https://arxiv.org/abs/2602.23097>.

## Наш BEM-CUDA: baseline для сравнения

Текущий проект `BEM-CUDA` - специализированный PMCHWT/RWG решатель под рассеяние света и массовые прогоны по ориентациям. По коду:

- CLI и маршрутизация режимов: `src/main.cpp`. Поддерживаются `sphere`, `hex_prism`, `--obj`, `--orient`, `--alpha-avg`, `--orient-start/count`, `--solver dense/fmm/pfft/spfft`, системы `pmchwt`, `balanced`, `muller`, `muller-balanced`, `muller2`, `muller2-balanced`, экспорт токов.
- Дискретизация: `src/mesh.cpp`, `src/rwg.cpp`, `src/quadrature.h`, `src/graglia.h`. RWG на треугольной поверхности, квадратуры 4/7/13, OBJ-нормировка к равновеликой сфере.
- PMCHWT assembly: `src/pmchwt.cu`, `src/assembly.cu`. Собираются exterior/interior L/K, затем блочная матрица `[eta_e L_e + eta_i L_i, -(K_e+K_i); K_e+K_i, L_e/eta_e + L_i/eta_i]`. Есть управляемая GPU-сборка: `BEM_ASM_GPU_LIST=4` закрепляет расчет за одной картой, `BEM_ASM_GPU_LIST=2,4` считает exterior/interior параллельно на двух картах. Для плотного solve и far-field усреднения аналогично используются `BEM_LU_GPU_LIST` и `BEM_FF_GPU_LIST`.
- Ускорители: `src/fmm.cu`, `src/bem_fmm.cu`, `src/p2p.cu`, `src/pfft.cu`, `src/surface_pfft.cu`. Основной рабочий путь - FMM+GMRES; pFFT/spFFT экспериментальны.
- Линейная алгебра: `src/gmres.cu`, `src/block_gmres.cu`, `src/precond.cu`, `src/solver.cu`. Есть dense LU, GMRES, block GMRES, автоматические/ручные прекондиционеры.
- Дальняя зона: `src/farfield.cu`; есть `FFCacheGPU`, пакетная обработка направлений/ориентаций, GPU packing коэффициентов, `alpha-avg` без пересборки системы.
- Ориентации и батчи: `src/orient.cpp`, `run_orient_mgpu.py`, `run_orient_queue.py`, `scripts/adaptive_orient_bem.py`.
- Контроль: Mie через `verify_mie.py`, сравнение с ADDA через `scripts/plot_bem_raw_adda.py`, постерные аудиты через `poster_a0/make_assets.py`.

Сильная сторона нашего кода - практическая специализация под одну задачу: много ориентаций, сравнение с ADDA/Mie, CUDA/FMM, GPU memory accounting, CLI для очередей. Слабая сторона - много экспериментальных режимов, сложная матрица параметров, слабая внешняя API-слойность и пока нерешенные ошибки слабых поляризационных элементов.

## 1. SCUFF-EM

### Архитектура кода

SCUFF-EM - зрелый C++ пакет Surface-CUrrent-Field Formulation of ElectroMagnetism. По структуре:

- `libs/libscuff`: ядро BEM; сборка матриц (`AssembleBEMMatrix*.cc`), RHS (`AssembleRHSVector.cc`), геометрия, surface-current representation, edge/panel interactions.
- `libs/libhmat`: H-matrix слой, dense/sparse matrix IO, LU/QR/eig wrappers, matrix-vector operations. Это важное отличие: SCUFF-EM имеет отдельную абстракцию матричного представления, а не только один FMM-matvec путь.
- `libs/libTriInt`: интегрирование по треугольникам, lattice sums, Lebedev, Brillouin-zone integration.
- `libs/libSubstrate`: layered/substrate Green functions, Sommerfeld integration, interpolation.
- `libs/libSpherical`: vector spherical waves и translation matrices.
- `libs/libIncField`: plane wave, Gaussian beam, point source, spherical wave.
- `applications`: набор CLI-программ поверх библиотеки.
- `tests/RunMieTest.sh`, `tests/RunFresnelTest.sh`, unit meshes: сильная проверка на аналитические задачи.

### Формализм и численная схема

- RWG/edge-based surface-current formulation.
- Поддерживает проводники, диэлектрики, сложные среды, периодические/слоистые задачи.
- Делает emphasis на правильных near-singular/singular интегралах edge-panel/panel-panel, что видно по interactive tests `TaylorDuffy`, `GetEPIs`, `QDFIPPI`, `QIFIPPI`.
- Есть H-matrix и отдельная инфраструктура для больших задач, но это CPU-oriented mature library, не специализированный CUDA-пакет для ориентационных sweep.

### Что лучше/хуже относительно нас

Лучше:

- Более чистая архитектура библиотеки: geometry, material, incident fields, matrix representation, applications разделены.
- Сильнее тестовая база: Mie/Fresnel tests, unit meshes, singular quadrature tests.
- Более развитая поддержка layered/substrate/periodic physics.
- H-matrix слой как альтернативный путь к памяти/скорости.

Хуже для нашей цели:

- Нет нашего CUDA/FMM production конвейера под сотни/тысячи ориентаций.
- Не заточен под ADDA-like Mueller-angle benchmarking и oldauto-style orientation averaging.
- Интеграция в наш pipeline потребует адаптации данных и нормировок.

### Что можно забрать

- Сделать отдельный test suite на Mie/Fresnel/symmetry и singular quadrature, не только скрипты в `runs`.
- Вынести incident-field/material/geometry API из `main.cpp` в отдельные модули.
- Изучить Taylor-Duffy/near-singular классификацию как источник для улучшения точности на острых ребрах и близких панелях.
- Рассмотреть H-matrix/ACA как memory fallback для больших частиц, где dense corrections или FMM workspace убивают память.

## 2. HELIOS

### Архитектура кода

HELIOS - C++/Python пакет HomogEneous and Layered medIa Optical Scattering. Код реально доступен в `mavrikak/Helios`; это не только статья. По дереву:

- `src/SIEFormPMCHW.cpp`, `include/SIEFormPMCHW.h`: PMCHWT формулировка, сборка 2N x 2N системы по RWG-базисам. Матрица заполняется потоками CPU: RWG группируются по опорным треугольникам, затем каждая группа считается против всех остальных.
- `src/SimJob.cpp`, `include/SimJob.h`: чтение job-файлов, создание плотной матрицы `2*EdgeCount x 2*EdgeCount`, LU через LAPACK `zgetrf_/zgetrs_`, альтернативный CGS/CGSQR из Fortran с Jacobi-precondition row scaling.
- `src/RWGFun.cpp`: стандартная half-RWG функция, знак задается ориентацией FRONT/BACK и геометрической проверкой относительно ребра/нормали/свободной вершины.
- `src/Domain*.cpp`, `include/Domain*.h`: домены для homogeneous, periodic homogeneous и layered media.
- `src/GreenFHom3D.cpp`, `GreenFHom3DPer*.cpp`, `GreenFLayered3D.cpp`: backends Green-функций для однородной, периодической и слоистой среды.
- `src/SommerfeldIntegrator.cpp`, `LayeredMediaUtils.*`: Sommerfeld-интегралы и таблицы/интерполяция для слоистых сред.
- `src/SingSub.cpp`: аналитические/полуаналитические сингулярные вычитания через line/surface integrals `K1--K4`.
- `src/SingCancellation.cpp`: singularity-cancellation квадратуры для пар треугольников с общей вершиной, ребром или гранью.
- `run_sie.py`, `pytools/jobwriter.py`, `pytools/meshconvert.py`, `pytools/visualization.py`, `helios_gui.py`: Python workflow для подготовки job, конвертации mesh, визуализации и GUI.
- `materials/*`, `sim_data/*`: таблицы материалов и готовые examples.

### Формализм и численная схема

- PMCHWT/RWG Galerkin SIE для проницаемых объектов.
- Явная модель доменов: треугольник хранит front/back domain indices, нормаль задает front-domain convention.
- Плотная матрица собирается на CPU; основная быстрая часть - многопоточная сборка по группам треугольников, а не FMM/GPU.
- Решение по умолчанию dense LU; итерационный режим есть, но это не production GMRES/FMM pipeline.
- Near/singular interactions обработаны подробнее, чем у нас сейчас: отдельные классы singular subtraction и singular cancellation.
- Есть layered/periodic physics, которой у нас практически нет.

### Сравнение с нами

HELIOS ближе к нашему проекту по физике, чем SCUFF-EM/Bempp: это light-scattering PMCHWT/RWG код, а не только универсальная BEM-библиотека. Но его вычислительный масштаб другой:

Лучше:

- Чистая архитектура: `Domain`, `GreenF`, `SIEForm`, `IncidentField`, `SurfaceMesh`, Python workflow разделены гораздо аккуратнее.
- Явная front/back-domain конвенция для нормалей и RWG знаков. Это полезно для наших проблем со слабыми Mueller-компонентами.
- Отдельный слой singular/near-singular интегрирования.
- Layered/periodic Green-function backends.
- Python GUI/tools вокруг C++ ядра, а не набор разрозненных shell/python scripts.

Хуже для нашей задачи:

- Нет CUDA/FMM matvec, нет multi-GPU production режима.
- Матрица плотная; память растет как O(N^2), LU как O(N^3), поэтому большие частицы и массовые ориентации будут хуже нашего CUDA/FMM пути.
- Нет нашего Mueller/ADDA/Mie benchmark pipeline и ориентационных batch-ускорений.
- Итерационный solve ограничен простым Jacobi scaling + CGSQR, без развитых предобуславливателей.

### Что можно забрать

- Front/back-domain convention как обязательная проверка mesh orientation и знаков RWG.
- Отдельный модуль singular cancellation для shared vertex/edge/facet pairs.
- Разделение `Domain`/`GreenF`/`SIEForm` у нас стоит повторить на уровне C++ API, не только CLI.
- Python workflow можно сделать похожим: один job object, один reproducible config, отдельные подготовка/solve/postprocess.
- Dense LU path HELIOS можно использовать как маленький reference-контроль для PMCHWT block signs, но не как быстрый расчетчик.
- Практический вывод для текущего кода: управлять устройствами явно, чтобы вычислительный эксперимент не зависел от GPU 0/1, и развивать отдельный near/singular слой вместо добавления новых ручных параметров solver-а.

## 3. Bempp-cl

### Архитектура кода

Bempp-cl - Python BEM library для Laplace/Helmholtz/Maxwell. В README заявлено, что C++ core прежнего Bempp заменен на JIT OpenCL kernels или Numba fallback. По дереву:

- `bempp_cl/api/operators`: high-level operators.
- `bempp_cl/api/space/maxwell_spaces.py`: Maxwell function spaces.
- `bempp_cl/api/assembly`: blocked operators, boundary operators, grid functions.
- `bempp_cl/core/opencl_kernels.py`, `opencl_assemblers.py`: JIT OpenCL assembly.
- `bempp_cl/core/numba_kernels.py`, `numba_assemblers.py`: CPU/JIT fallback.
- `bempp_cl/api/integration/duffy_galerkin.py`, `duffy_collocation.py`, `triangle_gauss.py`: singular/near-singular quadrature machinery.
- `bempp_cl/api/fmm/exafmm.py`, `fmm_assembler.py`: FMM integration through ExaFMM-style backend.
- `api/linalg/iterative_solvers.py`: iterative solve wrappers.

### Сравнение с нами

Bempp-cl сильнее как библиотека операторов: spaces, grid functions, blocked operators, API-level composition. Наш код сильнее как specialized executable: CUDA kernels, PMCHWT-specific packed farfield, Mueller output, orientation queues.

Bempp-cl полезен не как конкурент по скорости, а как архитектурный эталон:

- пространства и операторы разделены;
- blocked operator abstraction;
- Duffy quadrature как отдельный модуль;
- OpenCL/Numba JIT backends позволяют писать компактнее, но, вероятно, уступят hand-written CUDA для нашей узкой задачи.

### Что можно забрать

- Ввести `Operator`/`BlockedOperator` abstraction для PMCHWT/Muller/balanced систем вместо множества веток в `main.cpp`.
- Отдельно оформить function spaces: RWG, dual RWG/BC-подобные пространства, mass matrices.
- Сделать Duffy/Galerkin singular integration модульно и покрыть тестами.
- Добавить Python-level API, где задача собирается декларативно, а C++/CUDA исполняет тяжелые части.

## 4. MNPBEM

### Архитектура кода

MNPBEM - Matlab toolbox для metallic/dielectric nanoparticles, BEM approach Garcia de Abajo/Howie. По README и дереву:

- Matlab classes в `Base`, `BEM`, `Greenfun`, `Particles`, `Simulation`, `Material`.
- Есть retarded и quasistatic solvers: `bemret*`, `bemstat*`.
- Есть layer structures/substrates.
- Есть Mie solver (`Mie/miesolver.m`) для проверки сфер.
- Mesh tools (`Mesh2d`, `Particles`) и material database.
- Версия MNPBEM17 добавляет iterative solvers и H-matrices для нескольких тысяч/десятков тысяч boundary elements.
- MEX C++ backend для H-matrix/ACA-like Green table: `mex/hmatgreen*.cpp`, `mex/acagreen/*`, `mex/hlib/clustertree.h`.

### Сравнение с нами

MNPBEM зрелее в пользовательском научном workflow и нанофотонике. У него лучше оформлены materials/excitations/layers/particle shapes. Наш код быстрее и нижеуровневее для GPU/FMM и ориентационных усреднений; MNPBEM Matlab-oriented и не предназначен для массового GPU sweep по большим `ka`.

Точностно MNPBEM интересен:

- продуманная библиотека Green functions;
- отдельные near-field/far-field postprocessors;
- Mie validation встроена;
- H-matrix итерационный путь для крупных boundary elements.

### Что можно забрать

- Material/geometry/excitation как объектная модель, не CLI-флаги.
- H-matrix/ACA memory backend для случаев, где FMM дает плохую сходимость или много памяти.
- Систематический Mie-checker как first-class workflow, не отдельный ad-hoc скрипт.
- Mesh quality tooling: smoothing/refine/quality для сложных частиц.

## 5. nanobem22

### Архитектура кода

nanobem22 - Matlab toolbox для metallic and dielectric nanoparticles using Galerkin BEM. По дереву:

- `Boundary/Boundary.m`, `Material/Material.m` - явные boundary/material abstractions.
- `particleshapes/*` - генераторы треугольных форм: sphere, rod, polygon, cube, segment; есть edgeprofile.
- `Misc/integration/triquad.m`, `trisubdivide.m`, `triangle_unit_set.m` - интеграция по треугольникам.
- Help делит задачи на `stat` и `ret`, `quad`, `edge`, `mie`, `planewave`, `simulation`, `solution`.
- Есть Mie и vector spherical harmonic utilities.

### Сравнение с нами

nanobem22 ближе к хорошей “научной библиотеке” для малых/средних наночастиц. Он не заменяет наш CUDA/FMM backend, но показывает, как можно чисто оформить:

- particle/boundary/material abstractions;
- edge profiles и smooth particle generators;
- quadrature engine как отдельный reusable слой;
- help/tutorial documentation.

### Что можно забрать

- Edge-aware geometry generation: профиль ребра/скругление/контроль формы должен быть отдельным параметром, а не набором ручных mesh variants.
- Интеграционный движок с явной subdivision для near-singular cases.
- Набор эталонных частиц и `savegmsh`/mesh export workflow.

## 6. sekulicivan7/PMCHWT

### Архитектура кода

Это учебная/маленькая C++ реализация PMCHWT:

- `PMCHWT.cpp` читает `coord.txt`, `topol.txt`, `trian.txt`, строит mesh, собирает EFIE/MFIE exterior/interior, формирует 2N x 2N систему.
- `EFIEop.cpp`, `MFIEop.cpp` реализуют матричные элементы и singularity treatment.
- Решение dense через Eigen `colPivHouseholderQr()`.
- Нет FMM, нет GPU, нет ориентаций, нет farfield/Mueller pipeline.

### Сравнение с нами

Почти не конкурент, но полезен как компактный reference для знаков/масштабов PMCHWT:

```text
A11 = A1E + A2E
A12 = -A1M - A2M
A21 =  A1M + A2M
A22 = eta0^-2 A1E + eta2^-2 A2E
```

У нас похожая блочная структура в `src/pmchwt.cu`, но наша реализация уже масштабирована на GPU/FMM и разные нормировки систем.

### Что можно забрать

- Сделать минимальный CPU reference solver в нашем repo для маленьких сеток: одна функция без FMM/CUDA, dense exact assembly, чтобы ловить знаки PMCHWT и Mueller bugs.
- Использовать как шаблон для unit-теста PMCHWT block signs.

## 7. ScattPort / Code_Øyre / списки MoM

ScattPort - каталог программ light scattering, а не единый код. На странице Method of Moments перечислены, в частности, `for90-MoM2`, `code_Øyre`, `MoM_code_Oyre`, `BEM++ICE`, ACA Solver. Для `code_Øyre` ScattPort указывает Fortran code for electromagnetic scattering calculations for arbitrarily shaped closed surfaces using Method of Moments и ведет на NTNU Open. Это полезно как указатель на класс методов, но не как современная библиотека с ясным API, CI и репозиторием.

Сравнение с нами:

- Эти коды ближе к классическому dense/iterative MoM по произвольным замкнутым поверхностям, чем к нашему CUDA/FMM pipeline.
- Их главная практическая польза - не прямое заимствование кода, а набор эталонных постановок: произвольная поверхность, EFIE/MFIE/PMCHWT-подобные блоки, Fortran reference arithmetic.
- Для наших целей важнее SCUFF-EM/HELIOS/Bempp, потому что там есть явная структура RWG/SIE и современные near-singular modules.

Что можно забрать:

- В документации к каждому режиму явно указывать класс метода: dense MoM, accelerated SIE/FMM, DDA, T-matrix, физическая оптика.
- Не смешивать графики скорости разных классов без подписи области применимости.
- Использовать ScattPort как список внешних эталонов, когда нужны дополнительные сравнения, но не считать каталог проверенным исходным кодом.

## Сводное сравнение

| Проект | PMCHWT/RWG | Ускорение | GPU | Ориентационное усреднение | Light scattering output | Главная ценность для нас |
|---|---:|---|---:|---:|---:|---|
| BEM-CUDA | да | CUDA FMM/pFFT/экспериментальные batch modes | да | да, production scripts | Mueller, Mie/ADDA comparison | целевой production код |
| SCUFF-EM | да/родственный SIE | H-matrix, optimized CPU BEM | нет/не основной | нет как наш pipeline | scattering/RF/nanophotonics | зрелая архитектура, тесты, singular quadrature |
| HELIOS | да | CPU dense assembly, LU/CGSQR, singular cancellation | нет | нет как наш pipeline | cross sections, fields | близкая PMCHWT/RWG постановка, layered/periodic Green functions, чистая архитектура |
| Bempp-cl | Maxwell BIE/RWG spaces | OpenCL/Numba/JIT, FMM adapter | OpenCL | нет | general BIE | operator abstraction, Duffy quadrature |
| MNPBEM | BEM nanoparticles | H-matrix/iterative Matlab/MEX | нет | нет | cross sections, fields, Mie | materials/particles/H-matrix workflow |
| nanobem22 | Galerkin BEM | Matlab vectorization | нет | нет | nanophotonic spectra/fields | geometry/quadrature/edge profiles |
| PMCHWT toy | да | dense Eigen | нет | нет | solution coeffs only | minimal sign/reference solver |

## 10 вещей, которые реально могут помочь BEM-CUDA

1. **Минимальный dense CPU reference solver для PMCHWT.** Нужен в `tests/` для малых mesh: assemble L/K, solve, farfield, compare signs. Это поможет добить ошибки слабых Mueller-компонентов.

2. **SCUFF-style regression tests.** Добавить автоматические Mie/Fresnel/symmetry tests: sphere Mie, plane interface Fresnel, reciprocity, energy-like checks, M34=-M43 для диагональных амплитуд, invariance under rotation.

3. **Duffy/Taylor-Duffy near-singular module.** Сейчас точность на пыли/ребрах упирается в mesh/quad. Из SCUFF-EM/Bempp надо взять идею классификации pair interactions и отдельный near-singular path вместо общего quad4/7/13.

4. **Operator/BlockedOperator abstraction.** Убрать разрастание `--system pmchwt/muller/...` в процедурных ветках: сделать интерфейс для блоков `L`, `K`, mass/preconditioner и сборки систем.

5. **H-matrix/ACA fallback.** Для больших частиц и memory-kill нужен backend, который хранит приближенные блоки вместо дублирования плотных corrections/workspaces. Это не заменит FMM, но может быть контрольным и memory-safe режимом. Первый проверяемый шаг уже вынесен в `scripts/hmatrix_memory_audit.py`: он строит кластерное дерево по треугольникам, применяет admissibility criterion и оценивает `dense_full_gb`, `hmatrix_estimated_gb`, долю admissible блоков и потенциальное сжатие при заданном ранге ACA.

6. **Mesh quality gate перед расчетом.** Из MNPBEM/nanobem взять обязательные показатели: aspect ratio, min angle, edge-length distribution, curvature/edge tags. Запуск должен предупреждать, что mesh не годится для точности, до вычислений.

7. **Edge-aware meshing как отдельный workflow.** Не ручные `gmsh3400/4200/7000`, а параметры: target size on smooth faces, target size near edges, min angle, curvature adaptation, optional edge rounding. Это прямо бьет в проблему призмы/пылевой частицы.

8. **Python API поверх CLI.** HELIOS/Bempp/MNPBEM показывают, что workflow должен быть объектным: `Material`, `Shape`, `Mesh`, `Solver`, `OrientationGrid`, `Observable`. Наши scripts можно собрать в пакет `bemcuda/`.

9. **Разделить физическую нормировку и Mueller postprocess.** Ошибки слабых элементов подозрительно связаны с basis/sign/farfield conventions. Нужен отдельный модуль amplitude->Mueller с reference tests и экспортом S1/S2/S3/S4 до усреднения.

10. **Систематический orientation convergence driver.** Не только фиксированные сетки, а adaptive oldauto-like driver: накапливать ориентации, оценивать error bars по M11 и selected polarization components, останавливать по заданному критерию.

## Приоритет внедрения

Короткий порядок, который даст максимум пользы:

1. CPU reference PMCHWT + sign tests.
2. amplitude/Mueller audit tests.
3. mesh quality gate + edge-aware remeshing policy.
4. near-singular Duffy/Taylor-Duffy path.
5. Python API for reproducible production runs.
6. H-matrix/ACA memory fallback investigation.

## Что не стоит копировать напрямую

- Matlab-oriented architecture MNPBEM/nanobem не подходит для production CUDA.
- Universal Bempp abstractions могут замедлить горячий путь, если перенести их внутрь CUDA kernels. Их надо брать на уровне API/тестов, а не runtime matvec.
- SCUFF-EM layered/periodic physics пока не первоочередная, пока не закрыты слабые Mueller-компоненты и mesh strategy.
- HELIOS нельзя копировать как быстрый backend: плотная матрица и LU не подходят для наших больших ориентационных прогонов. Его стоит использовать как архитектурный и точностной ориентир: знаки RWG, домены, singular cancellation, Python workflow.
