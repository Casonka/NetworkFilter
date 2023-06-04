################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (10.3-2021.10)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
C:/Users/Caska/Desktop/projects/Python/embedded_filter/X-CUBE-AI/App/app_x-cube-ai.c \
C:/Users/Caska/Desktop/projects/Python/embedded_filter/X-CUBE-AI/App/network.c \
C:/Users/Caska/Desktop/projects/Python/embedded_filter/X-CUBE-AI/App/network_data.c \
C:/Users/Caska/Desktop/projects/Python/embedded_filter/X-CUBE-AI/App/network_data_params.c 

OBJS += \
./Application/User/X-CUBE-AI/App/app_x-cube-ai.o \
./Application/User/X-CUBE-AI/App/network.o \
./Application/User/X-CUBE-AI/App/network_data.o \
./Application/User/X-CUBE-AI/App/network_data_params.o 

C_DEPS += \
./Application/User/X-CUBE-AI/App/app_x-cube-ai.d \
./Application/User/X-CUBE-AI/App/network.d \
./Application/User/X-CUBE-AI/App/network_data.d \
./Application/User/X-CUBE-AI/App/network_data_params.d 


# Each subdirectory must supply rules for building sources it contributes
Application/User/X-CUBE-AI/App/app_x-cube-ai.o: C:/Users/Caska/Desktop/projects/Python/embedded_filter/X-CUBE-AI/App/app_x-cube-ai.c Application/User/X-CUBE-AI/App/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F401xC -c -I../../Core/Inc -I../../X-CUBE-AI/App -I../../X-CUBE-AI -I../../Middlewares/ST/AI/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Application/User/X-CUBE-AI/App/network.o: C:/Users/Caska/Desktop/projects/Python/embedded_filter/X-CUBE-AI/App/network.c Application/User/X-CUBE-AI/App/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F401xC -c -I../../Core/Inc -I../../X-CUBE-AI/App -I../../X-CUBE-AI -I../../Middlewares/ST/AI/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Application/User/X-CUBE-AI/App/network_data.o: C:/Users/Caska/Desktop/projects/Python/embedded_filter/X-CUBE-AI/App/network_data.c Application/User/X-CUBE-AI/App/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F401xC -c -I../../Core/Inc -I../../X-CUBE-AI/App -I../../X-CUBE-AI -I../../Middlewares/ST/AI/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Application/User/X-CUBE-AI/App/network_data_params.o: C:/Users/Caska/Desktop/projects/Python/embedded_filter/X-CUBE-AI/App/network_data_params.c Application/User/X-CUBE-AI/App/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F401xC -c -I../../Core/Inc -I../../X-CUBE-AI/App -I../../X-CUBE-AI -I../../Middlewares/ST/AI/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc -I../../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Application-2f-User-2f-X-2d-CUBE-2d-AI-2f-App

clean-Application-2f-User-2f-X-2d-CUBE-2d-AI-2f-App:
	-$(RM) ./Application/User/X-CUBE-AI/App/app_x-cube-ai.d ./Application/User/X-CUBE-AI/App/app_x-cube-ai.o ./Application/User/X-CUBE-AI/App/app_x-cube-ai.su ./Application/User/X-CUBE-AI/App/network.d ./Application/User/X-CUBE-AI/App/network.o ./Application/User/X-CUBE-AI/App/network.su ./Application/User/X-CUBE-AI/App/network_data.d ./Application/User/X-CUBE-AI/App/network_data.o ./Application/User/X-CUBE-AI/App/network_data.su ./Application/User/X-CUBE-AI/App/network_data_params.d ./Application/User/X-CUBE-AI/App/network_data_params.o ./Application/User/X-CUBE-AI/App/network_data_params.su

.PHONY: clean-Application-2f-User-2f-X-2d-CUBE-2d-AI-2f-App

