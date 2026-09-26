---
title: "What Is a Transparent Proxy? Linux TProxy in Practice"
description: "How transparent proxying works, where REDIRECT falls short, and how nftables and policy routing deliver traffic to a Linux TProxy listener."
translation_key: tproxy
date: 2024-07-27 14:29:22 +0800
translated_at: 2026-09-26
---

#### Introduction

Transparent proxying is both useful and interesting.

Because clients need no special configuration, it has many possible applications: managing multiple devices, controlling access, security protection, and load balancing.

Yet many people recognize the name without understanding how it works. Even people who have used transparent proxies for years may struggle to explain why their UDP configuration never quite works. This article first explains the concept, then examines the traditional REDIRECT approach and where it falls short. That limitation helps explain why Linux provides dedicated TProxy support, as well as the purpose of each line in the configurations that follow.

#### What is a transparent proxy?

A transparent proxy is a type of network proxy. It is called *transparent* because the client does not need to know that its traffic is being proxied.

Computer networks work in layers. After a client sends data, it does not control every step the packets take to reach the destination server.

Between the client sending a packet and that packet leaving our router, there are several points where we can intervene. If we intercept the packets we want and direct them to a proxy, we have implemented transparent proxying.

#### Transparent proxy support in Linux

Once we understand the idea, it is easy to imagine several ways to implement it.

There are tutorials for doing this with third-party software on Windows and Android. Here, however, I will focus on Linux's kernel support. Linux even has a facility called TProxy specifically for transparent proxying.

Before looking at TProxy, let us examine the traditional approach: using the REDIRECT target in iptables. In other words, transparent proxying is possible on Linux without TProxy.

Our tool is iptables, a command-line interface for configuring how the Linux kernel handles packets. We want it to do two things:

1. Select the packets that should go through the proxy. Finding the right place to intercept them requires some knowledge of Netfilter, which is outside this article's scope.
2. Direct those packets to the proxy.

For example, this command redirects UDP traffic destined for port 53 to a local proxy listening on port 1080:

```bash
iptables -t nat -A PREROUTING -p udp --dport 53 -j REDIRECT --to-port 1080
```

One command does it. If command-line syntax looks intimidating, compare it with the description above: apart from the syntax, it says almost exactly the same thing.

But there is a problem. The [Linux kernel documentation](https://docs.kernel.org/networking/tproxy.html#redirecting-traffic) describes serious limitations of this approach. REDIRECT changes the packet's destination address. A proxy that needs to discover the original UDP destination can therefore run into difficulties; recovering the original TCP destination can also be subject to races.

Why does this matter?

A UDP datagram carries port information inside an IP packet. To make a packet intended for a remote server arrive at the local proxy through REDIRECT, the destination is rewritten. The proxy then receives a packet addressed to itself. If it is supposed to relay the traffic onward, the rewritten destination alone no longer tells it where the client originally wanted to send it.

For TCP, a proxy can obtain the original destination through `SO_ORIGINAL_DST`, using connection-tracking information. That is a useful distinction, although it does not remove all the limitations identified in the kernel documentation. An older plain-language V2Ray guide also noted difficulties with its UDP transparent-proxy configuration and suggested proxying only TCP for basic browsing and video use.

TProxy offers a way to avoid destination rewriting. Let us configure it and use the example to understand how it works.

#### The TPROXY approach

Several components must cooperate:

1. **Kernel support.** In the OpenWrt setup used here, this means installing `kmod-nf-tproxy`.
2. **A packet-filtering tool with TProxy support.** The OpenWrt setup uses nftables rather than iptables. I learned the basics from scratch, too, so unfamiliar syntax need not be a barrier. On this setup, `kmod-nft-tproxy` supplies support for the nftables `tproxy` statement.
3. **Policy routing.** We will configure this below.
4. **A proxy application that supports TProxy.** It needs to use the appropriate transparent socket options; an ordinary proxy listener is not enough.

With those pieces available, we can reproduce the earlier example using TProxy.

First, write an nftables script:

```bash
#!/usr/sbin/nft -f

table inet proxy{
   chain output {
       type route hook output priority filter; policy accept;
       udp dport 53 meta mark set 1
   }
}
```

The central statement is `udp dport 53 meta mark set 1`. It has a simple meaning: mark UDP packets destined for port 53 with the value `1`. The mark distinguishes these packets from other traffic. The surrounding lines define the table, chain, and Netfilter hook.

Run the script:

```bash
chmod a+x script && ./script
```

Next, policy routing must make traffic addressed to remote destinations eligible for local delivery. This step is essential:

```bash
ip rule add fwmark 1 lookup 100
ip route add local 0.0.0.0/0 dev lo table 100
```

Some background helps here. Modern Linux routing-table identifiers are 32-bit values; historically the range was only 0–255. The identifiers `0`, `253`, `254`, and `255` have reserved meanings. Table `254` is the main routing table. The file `/etc/iproute2/rt_tables` contains table-name aliases, and you can add an alias for a custom table there.

The first command adds a policy rule that sends packets marked `1` to routing table `100`. There is nothing special about the number `100`: it is simply a convenient example, also used in the kernel documentation.

The second command adds a route to that table. The important part is `local 0.0.0.0/0 dev lo`: it treats every IPv4 destination as local for packets routed through this table, using the loopback interface.

This brings the selected locally generated traffic back into the local receive path.

Why is that essential? These packets were originally addressed to other machines. With ordinary outbound routing, they would leave the machine rather than return to the receive path where our PREROUTING TProxy rule can handle them. Policy routing makes delivery to a local proxy possible without changing their destination addresses.

> **Scope of this example:** the policy-routing commands above configure IPv4. The `inet` nftables table can match both IP families, but IPv6 interception also requires its own policy rule, local route, and compatible proxy listener.

Now select the marked packets and deliver them to the proxy through TProxy:

```bash
table inet proxy{
   chain prerouting{
       type filter hook prerouting priority mangle; policy accept;

       # Deliver the selected packets to the transparent proxy.
       udp dport 53 meta mark 1 tproxy to :1080
   }
}
```

This means: deliver UDP packets destined for port 53 and marked `1` to the proxy process listening on port 1080.

The important word is *deliver*. Unlike REDIRECT, `tproxy to :1080` does not rewrite the packet's destination address. Instead, the kernel delivers it to a suitable local socket on port 1080. The proxy uses a socket configured with `IP_TRANSPARENT`, allowing it to accept the intercepted traffic while preserving the original destination information.

Recall the limitation of REDIRECT: after rewriting, the destination fields no longer contain the client's original destination. TProxy avoids that loss of information in the packet itself. This also explains the requirement that the proxy application support TProxy: it must know how to use these special sockets.

That completes the basic mechanism.

For locally generated traffic, we used two chains. For traffic arriving from another machine, marking and interception can happen together in PREROUTING. Policy routing is still required:

```bash
udp dport 53 meta mark set 1 tproxy to :1080
```

#### Closing thoughts

This is a small example, but the same ideas extend to more complex configurations.

In practice, you may intercept a broad range of traffic rather than packets addressed to just one port or IP address. You must then exempt the proxy's own outbound traffic. Otherwise, its packets will be intercepted again as soon as they leave the proxy, creating a loop instead of reaching their destination.

There are several ways to distinguish that traffic: exempt the upstream destination address, apply a different mark to packets generated by the proxy, or run the proxy under a separate user and match that identity. nftables provides the building blocks for these approaches.

References:

- <https://docs.kernel.org/networking/tproxy.html>
- <https://powerdns.org/tproxydoc/tproxy.md.html>
- <https://xtls.github.io/document/level-2/transparent_proxy/transparent_proxy.html>
- <https://xtls.github.io/document/level-2/tproxy.html>
- <https://wiki.nftables.org/wiki-nftables/index.php/Matching_packet_metainformation>
- <https://man7.org/linux/man-pages/man8/ip-route.8.html>
- <http://git.netfilter.org/nftables/commit/?id=2be1d52644cf77bb2634fb504a265da480c5e901>
