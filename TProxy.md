# 透明代理

## 前言

从第一次听到“透明代理”这个概念就觉得非常有趣，花了好几天的时间学习了一下，觉得很满足，这里分享一下自己的收获。

## 什么是透明代理

首先透明代理是网络代理技术的一种，而之所以称它“透明”是因为客户端设备不知道自己的流量被代理了。我们知道在计算机网络中，数据的传递是分层的，客户端发出数据之后并不理会数据包如何到达目标服务器。于是在客户端发出数据之后，在数据包离开我们的路由器之前，中间还有许多环节我们可以控制和干预。如果我们中途拦截需要的数据包并且把它转发到代理服务器我们就实现“透明代理”。

## Linux的透明代理支持

了解了透明代理的概念之后我们，可以想象实现透明代理的方式是有很多的。网上能找到如何利用第三方软件在Windows, 安卓系统上实现透明代理的，可以自行了解。本文我们这里我们重点聊一聊Linux下的透明代理内核模块。是的，Linux下甚至有一个内核模块叫做TProxy专门用来支持透明代理。
不过在进入TProxy模块之前我们先来了解了一下传统的用iptables的REDIRECT语句来做到透明代理的方式。就是说哪怕没有TProxy这个模块，linux下也是可以实现透明代理的。
我们的工具是iptables。如果你不知道神马是iptables的话，它是一个命令行工具，能让你对linux内核处理数据包的方式进行一些定制。我们只需找出需要代理的数据包，用iptables的REDIRECT target把数据包转发到代理。这里举个例子，把去53端口的udp流量转发到运行在本地的1080端口的代理服务器：

`bash iptables -t nat -A PREROUTING -p udp --dport 53 -j REDIRECT --to-port 1080`

一条命令搞定，真正就和想的一样那么简单。
等等上边的语句好像并不能正常工作，linux内核开发文档中说这样实现透明代理有严重的缺陷，说这么做实际上改变了数据包的目标IP地址，导致UDP流量失去了原始目标地址，就算是TCP工作方式也不太正常。
怎么回事呢，原来UDP数据包就是非常简单地在ip数据包的基础上加上了端口，想要让本来去目标服务器的数据包去往代理服务器就得改掉ip数据包中的目标地址，但是把目标地址都改掉了，代理服务器也就无法正确地处理这个数据包了。TCP数据包因为存在握手机制和设计冗余，哪怕修改了TCP数据包，目标IP地址依然在IP报文中保存着，从而避免了这个问题。
怎么办呢，终于到了我们要讲的TProxy模块了，接下来我们就来实践一下，深入了解它的工作原理。

## TPROXY

要使用这种方式，需要多个组件的支持：

 1. 需要Linux内核的支持，比如我使用的是openwrt，需要安装kmod-nf-tproxy。
 2. 需要工具的支持，新版的openwrt使用nftables 而不是iptables，不过不要害怕，我也是从零开始学习了nftables的基本用法，再说网上iptables的资料已经够多了，我们就用点不一样的吧。想要nftables支持tproxy语句还需要安装kmod-nft-tproxy。
 3. 需要配合路由表使用。
 4. 需要你的代理服务器支持Tproxy.

准备好这些组件让我们看看它具体是怎么工作的。

还是上面的例子，我们写一个nftables脚本:  

 ```bash
 #!/usr/sbin/nft -f

 table inet proxy{
    chain output {
        type route hook output priority filter; policy accept;
        upd dport 53 meta set mark 1
    }
 }
 ```

虽然上面有这么大一坨，看着很复杂，实际上有用的就一句```upd dport 53 meta set mark 1```，而且它只做了一件非常简单的事儿，就是把去往53端口的UDP流量做了标记，把这些数据包挑出来备用。

然后我们需要路由表的配合，让这些明显不是去往本地的流量也能进入本地（很关键）， 运行下面的命令：

 ```bash
 ip rule add fwmark 1 lookup 100
 ip route add local 0.0.0.0/0 dev lo table 100
 ```
  
这里告诉大家一点小知识，Linux中有0-255，共256张路由表。其中0，253，254，255号路由表是特殊路由表，默认已经被配置了。如果不指定使用哪一张表就会默认使用254号表，也叫main. 可以用```cat cat /etc/iproute2/rt_tables```命令查看表的别名, 也可以修改这个文件，给自定义的表添加一个别名。

上面的语句首先增加一个策略把所有被标记过的的数据包都送到100号路由表。
然后我们给100号路由表添加一条路由规则。观察第二条命令 ```local 0.0.0.0/0```, 这条命令告诉内核这些所有的地址都属于本地地址，```dev lo```通过lo回环接口处理这些数据。

接下来我们把标记过的的数据包导入到代理：

 ```bash
 table inet proxy{
    chain prerouting{
        type filter hook output priority mangle; policy accept;
        udp dport 53 meta mark 1 tproxy to :1080
    }
 }
 ```

以上是本地数据我们需要两个链才能完成，如果不是本地数据， 只需要一条语句，把标记和导入到代理同时完成：

```bash
 udp dprot 53  mark set 1 tproxy to :1080
```

最后，一定要记得放行透明代理发出的数据包，否则数据包刚离开代理服务器又被拦截回来了，只能在本地回环，永远也发不出去了。至于放行的方法，其实有很多，两种常见的思路是要么给透明代理发出的数据包做上不同的标记，要么把代理运行在不同的用户下面加以区分。总之nftables非常强大，灵活运用即可。

参考链接：
 > <https://docs.kernel.org/networking/tproxy.html>
 > <https://powerdns.org/tproxydoc/tproxy.md.html>
 > <https://xtls.github.io/document/level-2/transparent_proxy/transparent_proxy.html>
 > <https://xtls.github.io/document/level-2/tproxy.html>
 > <https://wiki.nftables.org/wiki-nftables/index.php/Matching_packet_metainformation>
 > <https://man7.org/linux/man-pages/man8/ip-route.8.html>
 > <http://git.netfilter.org/nftables/commit/?id=2be1d52644cf77bb2634fb504a265da480c5e901>
