---
layout: default
title:  "什么是路由器？如何设置Archlinux作为软路由?"
date:   2025-02-25 00:13:22 +0800
categories: jekyll update
---
@import "{{ site.theme }}";


### 前言

在[前一篇博文](https://raycc.org/jekyll/update/2025/02/04/nanopi-r2s-arch-linux.html)中，我写了自己在NanopiR2s上安装ArchLinuxArm的经历，现在是时候用它做一些事儿了。

R2S非常适合做家庭软路由，所以我的目标是用把ArchLinux设置成路由器，当然最重要的是通过这个经历了解一些网络知识。

本文总结了设置路由器的过程，希望读者读完这篇博客之后对路由器的工作原理有深入的理解。

### 路由器

开始之前首先要了解一下什么是路由器。

相信每个人都至少见过路由器，家里的网络设备不管是有线还是无线都是要连接到路由器才能上网。

在家庭或公司等小型网络环境中，网络请求先发送到路由器，路由器再将这些请求转发给电信运营商。而运营商的路由器则负责将网络请求传递给其他网络，从而实现信息的流通与交互。

简而言之，路由器的核心功能就是连接不同的网络，确保数据能够在各个网络之间准确传输。

通常情况下，一个路由器至少需要两个网络接口，通过合理的配置，利用 Masquerading（伪装）的 NAT（网络地址转换）技术，结合 Linux 内核在不同接口间转发数据包的功能，就能实现多个网络的互联互通。

接下来就来看看具体设置步骤，深入理解它的工作原理。

### 安装步骤


#### 重命名接口
首先是一个可选的步骤，给网络接口重新命名，方便后续的配置。

在Linux系统中，设备由一个叫做udev的组件自动管理，其中当然也包括网络设备，即我们的网卡。Udev提供了一个可供其他组件调用的接口，也就是常说的网络接口。系统启动后由专门管理网络的软件Netmanager接手管理这些网络接口。 

以我的设备R2s为例子，它有两个网口，我把连接互联网的网卡对应的接口命名为ext, 把连接局域网的接口命名为int.

有多种方法可以更改网络接口配置。

我们可以更改udev的设置：

在/etc/udev/rules.d/文件夹中添加一个.link为后缀的文件，比如我们把它叫做10-network.rules。
然后添加以下内容：

```
SUBSYSTEM=="net", ACTION=="add", ATTR{address}=="", NAME="int"
SUBSYSTEM=="net", ACTION=="add", ATTR{address}=="", NAME="ext"
```
因为有两个接口，所以有两条语句。使用ip link show显示所有的接口信息，然后在ATTR{address}后面加上接口对应的Mac地址。

Udev会根据规则生成接口名称，当然也可以再启动之后用Netmanager修改。

举例来说，ArchLinuxArm默认使用systemd-networkd。添加配置文件/etc/systemd/network/10-ext.link：

```
[Match]
PermanentMACAddress=aa:bb:cc:dd:ee:ff #改成相应的mac地址

[Link]
Name=ext
```

注意修改udev配置文件或者systemd-networkd配置，两种方法任选其一即可。

#### 接口设置

然后要对两个接口分别进行设置。

不同的网络管理软件的配置文件写法不同，不过大体思路都一样：
1. 在连接局域网的接口上设置静态IP和DHCP服务，这样局域网的设备可以使用该接口作为网关设备并自动获取IP地址。   
2. 在连接互联网的接口上拨号或者设置成自动获取地址，再或者静态地址。一般路由器会直接连接光猫，保证路由器和光猫可以有效通信即可。  

还是以systemd-networkd为例子，分别创建或者修改两个接口的配置。

局域网接口配置文件：
```
[Match]
Name=in

[Network]
Address=10.0.0.1/24
DHCPServer=true
IPv4Forwarding=yes
IPv6Forwarding=yes
IPv6SendRA=yes
DHCPv6PrefixDelegation=yes

[DHCPServer]
PoolOffset=100
PoolSize=20
EmitDNS=yes
DNS=1.1.1.1
```
互联网接口配置文件：
```
[Match]
Name=out

[Network]
DHCP=yes
DNSSEC=no
IPv4Forwarding=yes
IPv6Forwarding=yes
IPv6PrivacyExtensions=true
```
命名为20-int.network 和20-ext.network，放到/etc/systemd/network/文件夹下。

配置文件的内容分为几个部分，Match部分选择我们上面设置的接口名称。也可以通过Mac地址匹配。在Network部分指定网络地址，HDCP或者其他的服务。最后在对应的部分对该服务进行进一步的配置。

其他网络管理软件的配置方法可参考下文参考链接Archwiki关于router的相关部分。

#### Masquerding

配置好两个接口之后还要开启内核转发功能，才能让数据包数据包通过不同的接口。在上面systemd-networkd的配置文件中有两个字段```"IPv4Forwarding=yes"```和 ```"IPv6Forwarding=yes"```,表示已经包括了允许内核转发数据包。

不过之开启转发功能还是不足以让局域网的设备连接上互联网，原因是局域网上的地址不会在互联网上传输。就算我们转发了网络请求，也还是会被的运营商的路由器丢弃。所以我们需要上面提到的Masquerading功能。

开启masqurading的方法非常简单，只需要在局域网的接口配置文件上加上正确地配置```"IPMasquerade="```。这个字段的默认参数是"no",只要改成"ipv4", "ipv6"或者"both"就可以了。

一旦开启了这个选项，甚至不需要显式开启IPv4Forwarding和IPv6Forwarding，而且会自动配置Nftables或者Iptables。

不过为了理解这个过程，我们自己配置Nftables。

写一个配置文件：
```
table inet nat {
  chain postrouting {
    type nat hook postrouting priority 0; policy accept;

    iif int oif ext masquerade
  }
}
```
可以看到自己实现也非常简单。只需注意Masqurading必要在postrouting这个钩子上指定，在匹配到需要操作的数据之后写上masqurade这个关键字就可以了。用```nft -f加载这个配置后就完成配置了。

当然到了这一步，你可能还是不太理解这个关键字干了什么，其实我们还可以换一个写法，写成```"iif int oif ext snat to ‘extIP’"```。意思是在Postrouting这个阶段，找出从局域网发出，去往互联网的数据包，把数据包的源地址改掉--从局域网地址改成有互联网连接的接口绑定的IP地址。

这样原本不能在互联网上转发的地址变成了可以使用的地址。

不过这样配置的问题是需要静态地址 ，或者提前在DHCP服务器上指定该接口的IP地址，然而我在上面的章节中只指定了面向局域网的接口的IP地址。这就是Masquerading派上用场的地方了，它在本质上和上面SNAT方法是一样的，唯一的区别是IP地址由内核自动指定。

到这里相信你已经理解了Masquerading这个技术。如果你看过我之前关于透明代理的[博客](https://raycc.org/jekyll/update/2024/07/27/TProxy.html)，相信你会觉得熟悉。有趣的是我们甚至可以用透明代理服务来代替Masquerading。

### 后续工作

到此为止，我的路由器就设置完毕了。插上试了试已经可以工作了。

为了安全性考虑，不应该在路由器上运行网络服务，它们应该运行在局域网的服务器上面。只需要使用DNAT把需要访问该地址的数据包目标地址改成相应的IP地址即可。

另外路由器暴露在互联网上面，设置防火墙是必要的步骤。因为是家庭路由，防火墙主要在Input链上，对从互联网上来的流量进行限制。

想要网络服务能正常工作还需要路由表等组件的配合使用，但是路由表是自动配置的，本文不多加赘述。希望读者已经对路由器有了更深刻的认识。



参考链接：
 > <https://wiki.archlinux.org/title/Router>  
 > <https://wiki.archlinux.org/title/Systemd-networkd>  
 > <https://man.archlinux.org/man/systemd.network.5>  
 > <https://wiki.nftables.org/wiki-nftables/index.php/Performing_Network_Address_Translation_(NAT)>  