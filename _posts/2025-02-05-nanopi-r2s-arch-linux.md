---
layout: post
title:  "Nanopi-R2s 刷arch linux的经历"
date:   2025-02-04 14:29:22 +0800
categories: jekyll update
---

### 我想装Arch Linux
最近我给自己的Nanopi-R2s上安装了Arch Linux。
如果用R2s做软路由，用OpenWRT官方固件就很好。厂商也提供了Ubuntu/Debian镜像。但是我偏爱Arch，因为对于没有相关知识的人来说很难。
安装Arch的过程很繁琐，没有一键工具，不过Arch提供非常全面和详细的文档。虽然过程具有挑战性，但它也提供了很好的支持，这是最佳的学习环境。
我从过去安装Arch Linux的过程中收益颇多。这次也想在我几年前买的Nanopi R2s上面试一试。  
网上虽然有一些教程，但是大多数都是提供操作步骤，如果不理解每个步骤的意义，出了问题就会很难处理。
我不打算也写一篇这样的教程，而是会对比PC架构，聊一下ARM系统的启动过程。
这篇文章是写给还没有刷Arch之前的自己，也适合那些想了解计算机启动过程的朋友。

### ARM 设备的启动流程
要成功部署系统，首先要理解设备从通电到系统就绪的全过程。我们将其简化为三个阶段：。
1. 固件初始化阶段： 
    - 传统PC：BIOS（Basic Input Output System）完成硬件初始化、POST自检、启动设备选择

    - ARM架构：SoC（System on chip）芯片内置BootROM，执行初始化后从存储设备固定位置加载引导程序

2. 引导加载阶段。加载系统的工具叫做Bootloader（引导程序）。它为硬件提供了一个抽象层，支持对复杂文件系统的操作，并且传递参数给内核。常见引导程序：
    * x86：GRUB
    * ARM：U-Boot（嵌入式设备主流选择）
    * Windows：bootmgfw.efi（UEFI环境）
1. 内核加载阶段。
这里存在一个"先有鸡还是先有蛋"的哲学问题：操作系统需要管理存储设备，但自身又存放在存储设备中。解决方案是：
   - Bootloader直接加载包含基础驱动和文件系统模块的内核
   - 通过initramfs（初始内存文件系统）建立临时根文件系统，最终挂载真正的根文件系统(rootfs)

基于上述原理，我们需要准备以下组件：

|组件	|作用	|获取方式
|------|------|-----
|U-Boot |	二级引导程序 |	交叉编译或预编译二进制
|Linux内核	| 系统核心	| 官方仓库或自定义编译
|设备树(Device Tree)	|硬件描述文件	|厂商SDK提供
|initramfs	| 临时根文件系统 | Arch Linux ARM官方镜像
|rootfs	|根文件系统	|Arch Linux ARM官方镜像


### 实战部署流程
参考[FriendlyElec Wiki](https://wiki.friendlyelec.com/wiki/index.php/NanoPi_R2S/zh#.E5.A6.82.E4.BD.95.E7.BC.96.E8.AF.91.E7.B3.BB.E7.BB.9F)配置交叉编译环境：
```bash
    # 安装友善提供的交叉编译器
    sudo bash -c \
  "$(curl -fsSL http://112.124.9.243:3000/friendlyelec/build-env-on-ubuntu-bionic/raw/branch/cn/install.sh)"
    #配置环境
    export PATH=/opt/FriendlyARM/toolchain/11.3-aarch64/bin:$PATH
    export GCC_COLORS=auto
```

我推荐从上游U-boot项目编译，而不是wiki上面uboot-rockchip。更简单直接，项目开发比较活跃。
```bash
    # 编译 ARM64 Rockchip SoC镜像
    git clone --depth 1 https://github.com/TrustedFirmware-A/trusted-firmware-a.git
    cd trusted-firmware-a
    make realclean
    make CROSS_COMPILE=aarch64-linux-gnu- PLAT=rk3328
    cd ..
    # 编译Uboot
    git clone --depth 1 https://source.denx.de/u-boot/u-boot.git
    cd u-boot
    export BL31=../trusted-firmware-a/build/rk3328/release/bl31/bl31.elf
    make evb-rk3328_defconfig
    make CROSS_COMPILE=aarch64-linux-gnu-
```
编译好之后把它刷入sc卡。因为SoC ROM不向BIOS有充足的空间，所以它的连接方式比较简单粗暴，就是从存储设备的固定位置读取。
微星瑞芯片遵循它自己的[分区标准](https://opensource.rock-chips.com/wiki_Partitions)。

| 阶段 |  名称      | 程序    | 文件    | 磁盘位置   
| ---  | -------  | -------  | -------      | ------- 
| 1      |  主引导   | ROM code | BootRom     |         
| 2      |  二级引导    | U-Boot   |  boot-rockchip.bin  | 0x40    
|        |                  | TPL/SPL  |             |         
| 3     |  boot分区 | Linux内核   | boot.img    | 0x8000  
|       |        |  Initrd镜像    |   initramfs-linux.img         
|       |        |  设备树二进制文件  |   rk3328-nanopi-r2s.dtb           
|       |        |       |      boot.scr       |         
| 4   |   root文件系统   |          |             |   0x40000   


请注意就像Bootloader不太够直接拉起系统一样，SoC ROM只能读取比较简单的程序，而U-boot相对于SoC显得像庞然大物，所以中间也有很多过度阶段。
这就是两步加载，甚至有三步加载。好消息是我们不用管这些，U-boot二进制程序里面包括第二步和第三步加载，直接把整个二进制文件刷到从64个扇区起的位置就可以了。

```dd if=u-boot-rockchip.bin of=/dev/sdX seek=64 conv=notrunc```

如果不行的，就需要分别制作TPL/SPL，Uboot和Trust的镜像，然后分别刷到第64，16384，24576扇区了,参考Uboot[文档](https://docs.u-boot.org/en/latest/board/rockchip/rockchip.html#package-the-image-with-rockchip-miniloader)。 

然后分区、刷文件系统，然后根据[ArchlinuxARM](https://archlinuxarm.org/platforms/armv8/rockchip/rock64)的指引下载和解压Rootfs文件。

```bash 
    # 为了简单，只分一个区
    parted /dev/sdX makpart '' ext4 32768s -1s
    mkfs.ext4 /dev/sdXp1
    mount /dev/sdXp1 /mnt
    wget http://os.archlinuxarm.org/os/ArchLinuxARM-aarch64-latest.tar.gz
    bsdtar -xpf ArchLinuxARM-aarch64-latest.tar.gz -C /mnt
```
接下来，Archwiki上我们下载一个Boot.scr的脚本，放在/boot文件下面。这个文件是连接U-boot 和Linux操作系统的关键步骤。
我们编译的U-boot里面有个配置参数CONFIG_DISTRO_DEFAULTS，如果允许它，Uboot会扫描可启动的磁盘里面的boot.scr或者extlinux.conf文件，然后执行这些文件。
一个extlinux.conf 文件看起来是这样：

``` extlinux.conf
label Arch with uart devicetree overlay
    kernel /arch/Image.gz
    initrd /arch/initramfs-linux.img
    fdt /dtbs/arch/board.dtb
    fdtoverlays /dtbs/arch/overlay/uart0-gpio0-1.dtbo
    append console=ttyS0,115200 console=tty1 rw root=UUID=fc0d0284-ca84-4194-bf8a-4b9da8d66908
```
这个配置文件告诉U-boot内核的位置，Initrd镜像的位置，FDT是关于设备的信息数据的二进制文件DTB的位置，然后是需要传递给内核的参数。
再看一下boot.scr脚本。需要注意的是boot.scr是一个二进制文件，我们不能直接打开，我们应该看对应的cmd文件。但是因为他们传递数据是基于文本，直接打开也能看出里面在干什么。
脚本里面内容比较多，我们不展示全文，只展示关键的部分。

```boot.scr
    setenv bootargs console=ttyS2,1500000 root=PARTUUID=${uuid}
    load ${devtype} ${devnum}:${bootpart} ${kernel_addr_r} /boot/Image
    load ${devtype} ${devnum}:${bootpart} ${fdt_addr_r} /boot/dtbs/${fdtfile};
    load ${devtype} ${devnum}:${bootpart} ${ramdisk_addr_r} /boot/initramfs-linux.img;
    booti ${kernel_addr_r} ${ramdisk_addr_r}:${filesize} ${fdt_addr_r};
```
可见这个文件里面就是一堆U-boot命令，而且作用和上面的extlinux.conf文件是一样的，即传递一些内核参数，加载内核，设备树二进制文件，以及Initrd镜像到内存的指定位置，最后启动内核和Initrd。
这些内存的位置则可以在源文件主板的Header文件中找到，比如rk3328_common.h里面的：
```
    #define ENV_MEM_LAYOUT_SETTINGS		\
	"scriptaddr=0x00500000\0"	\
	"script_offset_f=0xffe000\0"	\
	"script_size_f=0x2000\0"	\
	"pxefile_addr_r=0x00600000\0"	\
	"fdt_addr_r=0x01e00000\0"	\
	"fdtoverlay_addr_r=0x01f00000\0"	\
	"kernel_addr_r=0x02080000\0"	\
```
能不能直接下载Arch项目给的boot.scr文件？答案是不行，至少在我这里行不通。
Arch提供的initramfs镜像不能直接加载，需要编译成U-boot可以加载的形式。
dtb文件也需要修改成合适的，反正Arch提供的dtb文件在我这里没有一个能用的,需要自己编译内核和dtbs。
不用担心，只要知道这个文件的作用，我们完全可以自己写这个脚本。
我的boot.cmd脚本差不都是这样：
```
load ${devtype} ${devnum}:${distro_bootpart} ${ramdisk_addr_r} ${prefix}uInitrd
load ${devtype} ${devnum}:${distro_bootpart} ${kernel_addr_r} ${prefix}Image

load ${devtype} ${devnum}:${distro_bootpart} ${fdt_addr_r} ${prefix}dtbs/rk3328-nanopi-r2-rev00.dtb
fdt addr ${fdt_addr_r}
fdt resize 65536
booti ${kernel_addr_r} ${ramdisk_addr_r} ${fdt_addr_r}
```
写完之后，安装Uboot-tools包，然后用mkimage命令制作ramdisk和boot.scr:

``` 
    # 输入路径跟脚本中的地址对应
    mkimage -A arm -O linux -T ramdisk -C none -n "Initrd Image" -d /mnt/boot/initramfs-linux.img /mnt/boot/uInitrd;
    mkimage -A arm -O linux -T script -C none -n "Boot Script" -d boot.cmd /mnt/boot/boot.scr
```
想了解这个命令可以上网查询，这里就不详细讲了。
到了这一步就基本完成了，然后插电看看是不是已经好了。

### 必坑指南
通过USB-TTL模块查看启动日志。我没有这个模块，如果指示灯不显示好了，我也不知道问题出在那个环节，浪费了很多时间。
现在想来最好从Armbian这个项目下载一个能用的系统, 测试Uboot和boot.scr或者extlinux.conf能不能工作，最后再刷入Arch的文件系统了。
不建议使用友善官方镜像，因为它们分区太细了，需要debug的环节就太多了。
而且最好能准备两个sd卡，一个用于测试，一个用于正式部署。


参考链接：
><https://wiki.friendlyelec.com/wiki/index.php/NanoPi_R2S/zh> 
><https://opensource.rock-chips.com/wiki_Boot_option> 
><https://docs.u-boot.org/en/latest/board/rockchip/rockchip.html#rockchip-boards> 
><https://archlinuxarm.org/platforms/armv8/rockchip/rock64> 
><https://gist.github.com/larsch/a8f13faa2163984bb945d02efb897e6d>