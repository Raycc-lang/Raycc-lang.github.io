---
layout: post
title:  "Nanopi R2s 安装Arch Linux"
date:   2025-02-04 14:29:22 +0800
categories: jekyll update
---

# Nanopi-R2s 刷arch linux的经历

## 我想装Arch Linux
最近我给自己的Nanopi-R2s上安装了Arch Linux。
如果用R2s做软路由，用OpenWRT官方固件是非常好的选择。硬件提商也提供Ubuntu和Debian镜像。但是我偏爱Arch，因为它难。
安装Arch的过程很繁琐，没有一键工具，不过Arch提供非常全面和详细的文档。有挑战，但是也有很好的支持，这是最好的学习环境。
我从过去安装Arch Linux的过程中学到了很多东西。这次想把这个经验移植到ARM系统上。
网上虽然有一些教程，但是大多数只是提供安装步骤，如果不理解每个步骤的意义，出了问题就会很难处理。
我不打算也写一篇这样的教程，而是会对比x86系统，聊一下ARM系统的启动过程。
这篇文章是写给还没有刷Arch之前的自己，对于想到了解一点计算机启动过程的朋友也会有意义。

## ARM 设备的启动流程
首先了解计算机的启动过程，看它需要哪些组件，我们再逐一准备。
计算机启动的过程很复杂，但是为了便于理解我们把它简化成三个步骤:
1. BIOS(Bacia Input Output System). 这个是提前刷在主板上的，加电以后CPU会首先读取它，进行硬件初始化，POST（加电自检），然后寻找可启动设备。
Arm架构上没有BIOS,相对应的是SoC(System on Chip).
2. BIOS启动Bootloader。Bootloader是加载系统的工具。它为硬件提供了一个抽象层，支持对复杂文件系统的操作，并且传递参数给内核。
配合Linux系统常见的bootloader是Grub. ARM上用是U-boot。
bootmgfw.efi是 Windows 的 EFI 应用程序，负责加载内核（winload.efi）并处理启动配置（如恢复环境、多版本选择）。
3. 最后由Bootloader拉起系统。这个步骤不想想象中那么简单。
我们知道所有的计算机硬件包括磁盘和内存都是由操作系统管理的，但是操作系统本身也保存在文件系统之中，也需要加载到内存中。
所以在操作系统管理计算机之前就需要有一个操作系统了。比较简单的情况下Bootloader可以直接加载内核，内核里面已经包含了对文件系统的操作。
但是更多的情况下我们还需要一个中间步骤Initrd（Initial RAM Disk）或现代更常用的initramfs（Initial RAM Filesystem）。

知道了这个启动过程，然后我们就知道自己要准备什么：
1. 首先我们得有一个硬件，硬件包括了计算单元，SoC ROM。有的内置了eMMC(闪存)，没有的话得准备SD卡。我的是Nanopi-R2s。
2. 准备U-boot, 我们自己编译一个。有ARM64的Linux系统最好，没有的话安装一个交叉编译器。我用WSL配置的交叉编译环境。
3. 准备操作系统，包括内核和用户空间的基本配置。这个直接从官方下载。
4. 前面总结了三个步骤，但是这些步骤都比较简单，可以可以自己编译，也可以直接网上下载。
接下来，如何把每个环节连接起来，让它们成为一个可运行的整体是最重要的，这也是本文的意义所在。

## 实践
编译U-boot和内核可以直接看NanoPi R2S 
[wiki](https://wiki.friendlyelec.com/wiki/index.php/NanoPi_R2S/zh#.E5.A6.82.E4.BD.95.E7.BC.96.E8.AF.91.E7.B3.BB.E7.BB.9F)
我推荐从上有[U-boot项目](https://docs.u-boot.org/en/latest/build/gcc.html)编译，简单直接。项目开发比较活跃。
编译好之后把它刷入eMMC或者sc卡的固定位置。因为SoC ROM不向BIOS有充足的空间，所以它的连接方式比较简单粗暴，就是从存储设备的固定位置读取。
微星瑞芯片遵循它自己的[分区标准](https://opensource.rock-chips.com/wiki_Partitions)。
+--------+----------------+----------+-------------+---------+
| Boot   | Terminology #1 | Actual   | Rockchip    | Image   |
| stage  |                | program  |  Image      | Location|
| number |                | name     |   Name      | (sector)|
+--------+----------------+----------+-------------+---------+
| 1      |  Primary       | ROM code | BootRom     |         |
|        |  Program       |          |             |         |
|        |  Loader        |          |             |         |
|        |                |          |             |         |
| 2      |  Secondary     | U-Boot   |idbloader.img| 0x40    | pre-loader
|        |  Program       | TPL/SPL  |             |         |
|        |  Loader (SPL)  |          |             |         |
|        |                |          |             |         |
| 3      |  -             | U-Boot   | u-boot.itb  | 0x4000  | including u-boot and atf
|        |                |          | uboot.img   |         | only used with miniloader
|        |                |          |             |         |
|        |                | ATF/TEE  | trust.img   | 0x6000  | only used with miniloader
|        |                |          |             |         |
| 4      |  -             | kernel   | boot.img    | 0x8000  |
|        |                |          |             |         |
| 5      |  -             | rootfs   | rootfs.img  | 0x40000 |
+--------+----------------+----------+-------------+---------+
请注意就像Bootloader不太够直接拉起系统一样，SoC ROM只能读取比较简单的程序，而U-boot相对于SoC显得像庞然大物，所以中间也有很多过度阶段。
这就是两步加载，甚至有三步加载。但是我们不用管这些细节，好消息是我们编译的U-boot二进制程序里面包括第二步和第三步加载，
我们挂载SD卡，直接把整个二进制文件刷到从64个扇区起的位置就可以了。
```sudo dd if=u-boot-rockchip.bin of=/dev/sdX seek=64```
然后分区，刷文件系统，然后根据[ArchlinuxARM](https://archlinuxarm.org/platforms/armv8/rockchip/rock64)的指引下载和解压用户空间的文件。

```bash 
    wget http://os.archlinuxarm.org/os/ArchLinuxARM-aarch64-latest.tar.gz
    bsdtar -xpf ArchLinuxARM-aarch64-latest.tar.gz -C #挂载地址 
```
然后最关键的步骤来了，wiki上我们下载一个Boot.scr的脚本，放在/boot文件下面。这个文件是连接U-boot 和Linux操作系统的关键步骤。
我们编译的U-boot里面有个配置参数CONFIG_DISTRO_DEFAULTS，如果这个允许参数，Uboot会扫描可启动的磁盘里面的boot.scr或者更通用的extlinux.conf文件。然后执行这些文件。
一个extlinux.conf 文件看起来是这样：

``` extlinux.conf
    label Arch with uart devicetree overlay
        kernel /arch/Image.gz
        initrd /arch/initramfs-linux.img
        fdt /dtbs/arch/board.dtb
        fdtoverlays /dtbs/arch/overlay/uart0-gpio0-1.dtbo
        append console=ttyS0,115200 console=tty1 rw root=UUID=fc0d0284-ca84-4194-bf8a-4b9da8d66908
```
这个配置文件告诉U-boot内核的位置，我们上文提到的Initrd镜像的位置，FDT是关于设备的信息数据的二进制文件DTB的位置，然后则是需要传递给内核的参数。
再看一下boot.scr脚本。需要注意的是boot.scr是一个二进制文件，我们不能直接打开，我们应该看对应的cmd结尾的文件。但是因为他们传递数据是基于文本，直接打开也能看出里面在干什么。
脚本里面内容比较多，我们不展示全文，只展示关键的部分。

```boot.scr
    setenv bootargs console=ttyS2,1500000 root=PARTUUID=${uuid}
    load ${devtype} ${devnum}:${bootpart} ${kernel_addr_r} /boot/Image
    load ${devtype} ${devnum}:${bootpart} ${fdt_addr_r} /boot/dtbs/${fdtfile};
    load ${devtype} ${devnum}:${bootpart} ${ramdisk_addr_r} /boot/initramfs-linux.img;
    booti ${kernel_addr_r} ${ramdisk_addr_r}:${filesize} ${fdt_addr_r};
```
可见这个文件里面就是一堆U-boot命令，而且作用和上面的extlinux.conf文件是一样的，即传递一些内核参数，加载内核，设备树二进制文件，以及Initrd镜像到内存的指定位置，最后启动内核和Initrd。
这些内存的位置则可以在源文件主板的Header文件中找到，比如rk3328_common.h里面就有
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
说这么多有什么用吗，直接下载Arch项目给的boot.scr文件不行吗，答案是不行，至少在我这里行不通。
不用担心，我们作为高级用户，只要知道这个文件的作用，我们自己写这个脚本。
毕竟我们自己很清楚自己把这些文件放在哪个位置了，如需要传递哪些内核参数。
最坑的是，Arch提供的initramfs镜像不能直接加载，需要编译成U-boot可以加载的形式。
然后scr文件也需要编译，U-boot无法识别文本文件。
安装Uboot-tools包，然后

``` mkimage -A arm -O linux -T ramdisk -C none -n "Initrd Image" -d uInitrd.img /boot/initramfs-linux.img;
    mkimage -A arm -O linux -T script -C none -n "Boot Script" -d boot.cmd boot.scr
```
想了解这个命令可以上网查询，这里就不详细讲了。
到了这一步就基本完成了，然后插电看看是不是已经好了。
我自己是没有显示器，如果指示灯不显示好了，我也不知道问题出在那个环节，浪费了很多时间。
现在想来最好从Armbian这个项目下载一个能用的Ubuntu, 刷入自己编译的U-boot看能不能拉起Armbian, 
试试自己写的boot.scr或者extlinux.conf能不能工作，最后再刷入Arch的文件系统了。
不建议使用友善官方镜像，因为它们分区太细了，需要debug的环节就太多了。
而且最好能准备两个sd卡，一边看好没好，一边重新配置。

