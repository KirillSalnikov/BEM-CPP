import numpy as np
import os, math

Rad = math.pi/180.0

current_dir = os.getcwd()
folder_name = os.path.basename(current_dir)

theta = np.loadtxt('scattering_angles')
# Преобразование в вектор-строку
theta = theta.reshape(1, -1) 

ds = theta.copy()*0.0
N = ds.size
sum = 0.0
for l in range(0,N):
    if l==0:
        ds[0,l] = 2.0*math.pi*(1.0-math.cos(0.5*theta[0,1]*Rad))
    elif l==N-1:
        ds[0,l] = 2.0*math.pi*(1.0-math.cos(0.5*(theta[0,l]-theta[0,l-1])*Rad))
    else:
        ds[0,l] = 2.0*math.pi*(math.cos((theta[0,l]-0.5*(theta[0,l]-theta[0,l-1]))*Rad)-math.cos((theta[0,l]+0.5*(theta[0,l+1]-theta[0,l]))*Rad))
    sum = sum+ds[0,l]
print("norm =", sum) 
              
size = np.loadtxt('size_parameter')

# Список всех файлов
file_list = ['S11', 'S12', 'S13', 'S14', 
             'S21', 'S22', 'S23', 'S24',
             'S31', 'S32', 'S33', 'S34',
             'S41', 'S42', 'S43', 'S44']

# Загружаем все матрицы в словарь
matrices = {f: np.loadtxt(f) for f in file_list if os.path.exists(f)}

# Распаковываем словарь в отдельные переменные
locals().update(matrices)

#print("Доступные матрицы:", list(matrices.keys()))

headers = "theta 2pi*dcos " + " ".join(file_list)
headers_norm = " ".join([f + "_norm" for f in file_list[1:]])
full_headers = headers + " " + headers_norm 
print(full_headers)

i = 0
for x in size:
    file = os.path.join(current_dir, f'A_x={x}_{folder_name}.dat') 
    with open(file, 'w') as outfile:
        # Формируем исходную матрицу
        matrix_rows = [theta[0]] + [ds[0]] + [matrices[f][i,:] for f in file_list]
        full_matrix = np.array(matrix_rows).transpose()

        # Нормировка всех столбцов на S11
        first_column = full_matrix[:, 2].reshape(-1, 1)

        # Избегаем деления на ноль
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_cols = np.divide(full_matrix[:, 3:], first_column, 
                                       where=first_column!=0, 
                                       out=np.zeros_like(full_matrix[:, 3:]))

        # Объединяем исходную матрицу с нормированными столбцами
        full_matrix_with_norm = np.column_stack([full_matrix, normalized_cols])

        # Сохраняем в файл
        np.savetxt(file, full_matrix_with_norm, delimiter=' ', fmt='%g', header=full_headers, comments='')
        i = i+1
            
LR = np.loadtxt('LR')
DR = np.loadtxt('PLDR')

S11_theta180 = matrices['S11'][:,-1]
S22_theta180 = matrices['S22'][:,-1]
back_matrix = np.column_stack([size, S11_theta180, S22_theta180, LR, DR])
np.savetxt(os.path.join(current_dir, f'A_back_{folder_name}.dat'), back_matrix, delimiter=' ', fmt='%g', header='x S11 S22 LR PLDR', comments='')
