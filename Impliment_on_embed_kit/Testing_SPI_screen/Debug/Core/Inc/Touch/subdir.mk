################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (12.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/Inc/Touch/Touch.c 

C_DEPS += \
./Core/Inc/Touch/Touch.d 

OBJS += \
./Core/Inc/Touch/Touch.o 


# Each subdirectory must supply rules for building sources it contributes
Core/Inc/Touch/%.o Core/Inc/Touch/%.su Core/Inc/Touch/%.cyclo: ../Core/Inc/Touch/%.c Core/Inc/Touch/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F401xE -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-Inc-2f-Touch

clean-Core-2f-Inc-2f-Touch:
	-$(RM) ./Core/Inc/Touch/Touch.cyclo ./Core/Inc/Touch/Touch.d ./Core/Inc/Touch/Touch.o ./Core/Inc/Touch/Touch.su

.PHONY: clean-Core-2f-Inc-2f-Touch

