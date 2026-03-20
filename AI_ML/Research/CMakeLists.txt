# CMakeList.txt: projekt CMake dla elementu Research, dołącz źródło i zdefiniuj
# tutaj logikę specyficzną dla projektu.
#
cmake_minimum_required (VERSION 3.17)

# Włącz przeładowywanie kodu XAML na gorąco dla kompilatorów MSVC, jeśli jest to obsługiwane.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("Research" LANGUAGES CXX CUDA)

# CUDA - ustawienia globalne
enable_language(CUDA)
if (CMAKE_VERSION VERSION_GREATER 3.12)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Znalezienie pakietów PRZED utworzeniem targetu
find_package(Torch REQUIRED)
find_package(CUDAToolkit)

add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE ${CMAKE_SOURCE_DIR}/vendor)

# Dodaj źródło do pliku wykonywalnego tego projektu
add_executable (Research "Research.cpp" "Research.h" "src/data/ImageDataset.h" "src/data/ImageDataset.cpp" "src/models/ModelLoader.h" "src/models/ModelLoader.cpp" "src/training/Trainer.h" "src/training/Trainer.cpp" "src/data/ImageDatasetConfig.h" "src/training/metrics.h" "src/models/Classifier.h" "src/training/metrics.cpp" "src/models/Classifier.cpp")

# Ustawienia CXX_STANDARD dla targetu (PO jego utworzeniu)
if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Research PROPERTY CXX_STANDARD 20)
endif()

# Linkowanie libtorch
target_link_libraries(Research "${TORCH_LIBRARIES}")

# Linkowanie CUDA (jeśli dostępny)
if (CUDAToolkit_FOUND)
  target_link_libraries(Research CUDA::cudart CUDA::cuda_driver)
endif()

# Ścieżki include
target_include_directories(Research PRIVATE ${TORCH_INCLUDE_DIRS})

# Ustawienia specyficzne dla MSVC
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET Research
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:Research>)
endif (MSVC)