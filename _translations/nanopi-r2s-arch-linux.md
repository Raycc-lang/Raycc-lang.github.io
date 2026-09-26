---
title: "How Does a Computer Boot? Installing Arch Linux ARM on a NanoPi R2S"
description: "Compare PC and ARM boot sequences, then follow the U-Boot, root filesystem, and boot script setup for Arch Linux ARM on a NanoPi R2S."
translation_key: nanopi-r2s-arch-linux
date: 2025-02-04 14:29:22 +0800
translated_at: 2026-09-26
---

#### Why I wanted to install Arch Linux

I recently installed Arch Linux on my NanoPi R2S. Here are the lessons I learned along the way. This article is for anyone curious about how a computer boots, and it should also be useful if you want to install Arch Linux ARM on an ARM device.

Installing Arch Linux ARM on hardware without official support can be difficult for a beginner. It is also an opportunity to understand the lower layers of a Linux system. That is part of Arch's value: it makes you deal directly with the underlying configuration.

There are tutorials online, but most provide only a sequence of steps. If you do not understand what each step does, troubleshooting becomes difficult. Instead of writing another recipe like that, I will compare ARM and PC boot sequences, then explain the installation process in detail.

#### How an ARM device boots

To install a working system, we first need to understand what happens between powering on the device and having a usable operating system. We can simplify this into three stages:

1. **Firmware initialization**
   - On a traditional PC, the BIOS (Basic Input/Output System) initializes hardware, performs the power-on self-test, and selects a boot device.
   - On the ARM system discussed here, the SoC (system on a chip) contains a BootROM. After initialization, it loads a bootloader from a fixed location on the storage device.

2. **Bootloader execution**
   - A bootloader loads the operating system. It provides a layer of abstraction over the hardware, supports more complex filesystems, and passes parameters to the kernel.
   - Common examples include GRUB on x86 systems, U-Boot on embedded ARM devices, and Windows' `bootmgfw.efi` in a UEFI environment.

3. **Kernel loading**
   - There is a chicken-and-egg problem here: the operating system needs to manage the storage device, but the operating system itself lives on that device.
   - The bootloader loads a kernel with the basic drivers and filesystem support needed to start up.
   - An initramfs (initial RAM filesystem) provides a temporary root filesystem during startup, before the real root filesystem, or rootfs, is mounted.

Based on this process, we need the following components:

| Component | Purpose | Where to get it |
| --- | --- | --- |
| U-Boot | Secondary bootloader | Cross-compile it or use a prebuilt binary |
| Linux kernel | Core of the operating system | Official repositories or a custom build |
| Device tree | Hardware description | Vendor SDK or a custom build |
| initramfs | Temporary root filesystem | Arch Linux ARM image |
| rootfs | Root filesystem | Arch Linux ARM image |

#### Installing the system

The installation consists of four tasks: compile U-Boot, write it to the required disk location, create a partition and extract the root filesystem, and use `boot.scr` to connect U-Boot to the kernel. In terms of the three stages above, the first two tasks prepare the bootloader stage, and the last two prepare the kernel-loading stage. The BootROM is built into the chip; we neither need nor get to change it.

Set up the cross-compilation environment using the [FriendlyElec wiki](https://wiki.friendlyelec.com/wiki/index.php/NanoPi_R2S/zh#.E5.A6.82.E4.BD.95.E7.BC.96.E8.AF.91.E7.B3.BB.E7.BB.9F):

```bash
    # Install the cross-compiler provided by FriendlyElec.
    # This URL uses unencrypted HTTP and a bare IP address.
    # Download and inspect the script with curl before running it with sudo.
    sudo bash -c \
  "$(curl -fsSL http://112.124.9.243:3000/friendlyelec/build-env-on-ubuntu-bionic/raw/branch/cn/install.sh)"
    # Configure the environment.
    export PATH=/opt/FriendlyARM/toolchain/11.3-aarch64/bin:$PATH
    export GCC_COLORS=auto
```

First, build U-Boot. I recommend building from the upstream U-Boot project instead of the `uboot-rockchip` repository used in the wiki. I found the upstream route more straightforward, and the project is actively developed.

```bash
    # Build the firmware for the ARM64 Rockchip SoC.
    git clone --depth 1 https://github.com/TrustedFirmware-A/trusted-firmware-a.git
    cd trusted-firmware-a
    make realclean
    make CROSS_COMPILE=aarch64-linux-gnu- PLAT=rk3328
    cd ..
    # Build U-Boot.
    git clone --depth 1 https://source.denx.de/u-boot/u-boot.git
    cd u-boot
    export BL31=../trusted-firmware-a/build/rk3328/release/bl31/bl31.elf
    make evb-rk3328_defconfig
    make CROSS_COMPILE=aarch64-linux-gnu-
```

Once the build is complete, write the image to the SD card. Unlike a PC BIOS, the SoC's ROM has very limited space. Its way of finding the next stage is therefore quite direct: it reads from a fixed location on the storage device.

Rockchip devices follow their own [partition layout](https://opensource.rock-chips.com/wiki_Partitions):

| Stage | Name | Program or component | File | Disk location |
| --- | --- | --- | --- | --- |
| 1 | Primary boot stage | ROM code | BootROM | |
| 2 | Secondary boot stage | U-Boot | `u-boot-rockchip.bin` | `0x40` |
| | | TPL/SPL | | |
| 3 | Boot partition | Linux kernel | `boot.img` | `0x8000` |
| | | Initrd image | `initramfs-linux.img` | |
| | | Device tree binary | `rk3328-nanopi-r2s.dtb` | |
| | | Boot script | `boot.scr` | |
| 4 | Root filesystem | | | `0x40000` |

There are intermediate stages here, too. Just as a temporary root filesystem helps bridge the gap between loading the kernel and using the installed system, smaller bootloader stages bridge the gap between the SoC ROM and the full U-Boot program. The ROM can load only a relatively simple program; full U-Boot is much larger.

This is often a two-stage or even three-stage loading process: TPL/SPL starts first, then loads a larger bootloader. Fortunately, the combined U-Boot binary includes these later stages, so we can write the whole image starting at sector 64:

```bash
dd if=u-boot-rockchip.bin of=/dev/sdX seek=64 conv=notrunc
```

Why sector 64? Look back at the partition table: the secondary boot stage starts at `0x40`, which is 64 in decimal. That is where `seek=64` comes from. At this stage, the BootROM does not interpret the partition table or filesystem; it reads from the agreed location. That offset must be correct.

If the combined image does not work, the alternative is to prepare separate TPL/SPL, U-Boot, and Trust images and write them at sectors 64, 16384, and 24576, respectively. See the [U-Boot documentation](https://docs.u-boot.org/en/latest/board/rockchip/rockchip.html#package-the-image-with-rockchip-miniloader).

Next, create a partition and filesystem, then download and extract the root filesystem following the [Arch Linux ARM instructions](https://archlinuxarm.org/platforms/armv8/rockchip/rock64).

```bash
    # Use a single partition to keep things simple.
    parted /dev/sdX mkpart '' ext4 32768s -1s
    # Create an ext4 filesystem.
    mkfs.ext4 /dev/sdX1
    # Mount the partition.
    mount /dev/sdX1 /mnt
    # Download and extract the Arch root filesystem.
    wget http://os.archlinuxarm.org/os/ArchLinuxARM-aarch64-latest.tar.gz
    bsdtar -xpf ArchLinuxARM-aarch64-latest.tar.gz -C /mnt
```

Next, download a `boot.scr` script from the Arch wiki and put it in `/boot`. This file is the connection between U-Boot and the Linux operating system.

The U-Boot configuration used here has an option called `CONFIG_DISTRO_DEFAULTS`. When enabled, it lets U-Boot scan bootable storage for a `boot.scr` script or an `extlinux.conf` configuration and use it to boot.

An `extlinux.conf` file looks like this:

```text
label Arch with uart devicetree overlay
    kernel /arch/Image.gz
    initrd /arch/initramfs-linux.img
    fdt /dtbs/arch/board.dtb
    fdtoverlays /dtbs/arch/overlay/uart0-gpio0-1.dtbo
    append console=ttyS0,115200 console=tty1 rw root=UUID=fc0d0284-ca84-4194-bf8a-4b9da8d66908
```

This configuration tells U-Boot where to find the kernel, the initrd image, and the FDT (flattened device tree), a binary description of the hardware. It also supplies the parameters to pass to the kernel.

Now consider `boot.scr`. It is a binary file, so its corresponding `.cmd` source is the appropriate file to read or edit. Because its payload consists of text commands, however, opening the binary can still give you an idea of what it does.

The full script is longer; these are the key lines:

```text
    setenv bootargs console=ttyS2,1500000 root=PARTUUID=${uuid}
    load ${devtype} ${devnum}:${bootpart} ${kernel_addr_r} /boot/Image
    load ${devtype} ${devnum}:${bootpart} ${fdt_addr_r} /boot/dtbs/${fdtfile};
    load ${devtype} ${devnum}:${bootpart} ${ramdisk_addr_r} /boot/initramfs-linux.img;
    booti ${kernel_addr_r} ${ramdisk_addr_r}:${filesize} ${fdt_addr_r};
```

The script is a sequence of U-Boot commands. It serves the same purpose as `extlinux.conf`: set kernel parameters, load the kernel, device tree binary, and initrd into specified memory locations, then start the kernel with the initrd.

The variables for these memory locations are defined in board-related headers in the source tree. For example, `rk3328_common.h` contains:

```c
    #define ENV_MEM_LAYOUT_SETTINGS		\
	"scriptaddr=0x00500000\0"	\
	"script_offset_f=0xffe000\0"	\
	"script_size_f=0x2000\0"	\
	"pxefile_addr_r=0x00600000\0"	\
	"fdt_addr_r=0x01e00000\0"	\
	"fdtoverlay_addr_r=0x01f00000\0"	\
	"kernel_addr_r=0x02080000\0"	\
```

The downloaded `boot.scr` did not work unchanged for me. I encountered at least two problems:

1. **Arch supplies the initramfs as a raw image.** A raw initrd can be loaded, but the `booti` command must explicitly include its size, using `${ramdisk_addr_r}:${filesize}` as in the Armbian script above. If you do not want to pass `:size` on the `booti` line, you can use `mkimage` to add a header so U-Boot can read the size from the image itself. I chose the latter approach.
2. **The device tree must match the hardware and kernel.** The NanoPi R2 family has multiple revisions, such as `rev00` and `rev01`, with hardware differences and corresponding device trees. Filenames also vary between kernel sources: mainline Linux uses `rk3328-nanopi-r2s.dtb`, while FriendlyElec's BSP tree uses names such as `rk3328-nanopi-r2-rev00.dtb`. None of the prebuilt Arch device trees worked in my setup; I eventually resolved this by compiling the kernel and device trees myself. The filename in the `boot.cmd` below is simply the one that worked for that installation. Check the actual name of your device tree before copying it.

Once we understand what the script does, we can write our own. My `boot.cmd` looked roughly like this:

```text
load ${devtype} ${devnum}:${distro_bootpart} ${ramdisk_addr_r} ${prefix}uInitrd
load ${devtype} ${devnum}:${distro_bootpart} ${kernel_addr_r} ${prefix}Image

# The DTB filename depends on the kernel source and board revision.
# This installation used the FriendlyElec BSP's rev00 file; see the explanation above.
load ${devtype} ${devnum}:${distro_bootpart} ${fdt_addr_r} ${prefix}dtbs/rk3328-nanopi-r2-rev00.dtb
fdt addr ${fdt_addr_r}
fdt resize 65536
booti ${kernel_addr_r} ${ramdisk_addr_r} ${fdt_addr_r}
```

After writing the script, install the U-Boot tools package and use `mkimage` to create the ramdisk image and `boot.scr`:

```bash
    # Keep these paths consistent with the paths used in the script.
    mkimage -A arm64 -O linux -T ramdisk -C none -n "Initrd Image" -d /mnt/boot/initramfs-linux.img /mnt/boot/uInitrd;
    mkimage -A arm64 -O linux -T script -C none -n "Boot Script" -d boot.cmd /mnt/boot/boot.scr
```

What `mkimage` does is simple: it wraps the file in a header that U-Boot understands. The header records the image type, architecture, size, and checksum. When U-Boot loads `uInitrd` or `boot.scr`, it can identify the image, determine its size, and check its integrity.

This also answers the earlier question about why Arch's initramfs could not be used directly in my script. Raw images are not inherently unloadable. The issue was that my `booti` line did not include `:size`, unlike the Armbian example's `${ramdisk_addr_r}:${filesize}`. That is the difference between the two forms: either supply the size in the command or embed it in a `mkimage` header. I chose the latter, so my script could use `booti ${kernel_addr_r} ${ramdisk_addr_r} ${fdt_addr_r}` without `:size`, at the cost of wrapping the initramfs first.

At this point, U-Boot is in place as the bootloader and the setup is largely complete. Power on the device and see whether it boots.

#### Lessons from troubleshooting

Use a USB-to-TTL serial adapter to read the boot logs. I did not have one and could only watch the indicator lights. When something failed, I could not tell which stage had failed, and I wasted a lot of time.

The device tree was the easiest place to get stuck. The NanoPi R2 family has multiple revisions, filenames vary between kernel sources, and the prebuilt Arch files did not work for me. Without serial output, a failure at this stage is almost impossible to diagnose systematically. I suggest starting with a working device tree from an Armbian image to establish a working boot sequence, then switching to the Arch root filesystem.

In hindsight, I would first download a working Armbian system and verify that U-Boot and `boot.scr` or `extlinux.conf` work, then replace the root filesystem with Arch's.

I would avoid FriendlyElec's official images for this approach: their more elaborate partition layout adds too many pieces to debug.

If possible, prepare two SD cards: one for experiments and one for the final installation.

References:

- <https://wiki.friendlyelec.com/wiki/index.php/NanoPi_R2S/zh>
- <https://opensource.rock-chips.com/wiki_Boot_option>
- <https://docs.u-boot.org/en/latest/board/rockchip/rockchip.html#rockchip-boards>
- <https://archlinuxarm.org/platforms/armv8/rockchip/rock64>
- <https://gist.github.com/larsch/a8f13faa2163984bb945d02efb897e6d>
