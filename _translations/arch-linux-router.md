---
title: "What Is a Router? Setting Up Arch Linux as a Software Router"
description: "Configure LAN and WAN interfaces with systemd-networkd, then explore packet forwarding and NAT masquerading with nftables."
translation_key: arch-linux-router
date: 2025-02-25 00:13:22 +0800
translated_at: 2026-09-26
---

#### Introduction

In my [previous article (Chinese)]({% post_url 2025-02-05-nanopi-r2s-arch-linux %}), I described installing Arch Linux ARM on a NanoPi R2S. Now it is time to put it to work.

The R2S is well suited to being a home software router. My goal is to configure Arch Linux for that role and, more importantly, use the process to understand networking better.

This article walks through that configuration with the aim of explaining how a router works.

#### What is a router?

Before configuring one, we need to understand what a router does.

Most of us have encountered a home router. Our wired and wireless devices connect through it to reach the Internet.

In a small home or office network, traffic first reaches the router, which passes it toward the Internet service provider. The router connects the local network to the provider's network. The provider's routers then carry traffic onward to other networks.

The router's central job is therefore to connect networks. It typically has at least two network interfaces and forwards packets between them according to rules stored in a routing table.

The Linux kernel already supports forwarding packets using routing tables. Enabling forwarding turns an ordinary computer into a router implemented in software. For a typical home IPv4 network, however, forwarding alone is not enough: the LAN uses private addresses that cannot receive replies through the public Internet directly. We also need NAT masquerading to replace each outgoing packet's source address with the address of the outgoing interface.

Let us see how the configuration fits together.

#### Configuration steps

There are three main steps: optionally give the two interfaces meaningful names, configure the internal and external interfaces, then enable forwarding and address masquerading. Each step corresponds to part of the router's job described above.

##### Renaming the interfaces

First, an optional step: rename the interfaces to make the remaining configuration easier to follow.

On Linux, udev identifies devices and participates in naming network interfaces, such as `eth0`, when they appear. Network management software, such as systemd-networkd in this example, then configures those interfaces.

My R2S has two Ethernet ports. I call the Internet-facing interface `ext` and the LAN-facing interface `int`.

There are several ways to assign interface names. One is to add udev rules in a file ending in `.rules` under `/etc/udev/rules.d/`, for example `10-network.rules`:

```text
SUBSYSTEM=="net", ACTION=="add", ATTR{address}=="", NAME="int"
SUBSYSTEM=="net", ACTION=="add", ATTR{address}=="", NAME="ext"
```

There is one rule per interface. Run `ip link show` to inspect the interfaces, then put the corresponding MAC address inside each empty `ATTR{address}` value.

Another option on a system using systemd's network device configuration is a `.link` file. For the external interface, create `/etc/systemd/network/10-ext.link`:

```ini
[Match]
# Replace this with the corresponding MAC address.
PermanentMACAddress=aa:bb:cc:dd:ee:ff

[Link]
Name=ext
```

Use the same approach for the internal interface, with its own MAC address and `Name=int`. A `.link` file is applied through udev when the device is initialized; it is not a live interface-renaming command issued by networkd.

Choose either custom udev rules or `.link` files for naming; you do not need both.

##### Configuring the interfaces

Next, configure each interface for its role.

The configuration syntax varies between network managers, but the idea is the same:

1. Give the LAN interface a static IP address and enable a DHCP server. LAN devices can then obtain their addresses automatically and use this interface as their gateway.
2. Configure the Internet-facing interface to obtain an address automatically, use a static address, or establish the connection required by the provider. A home router often connects to an optical network terminal or modem; that upstream connection must work first.

Using systemd-networkd, create the following LAN configuration:

```ini
[Match]
Name=int

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

The Internet-facing configuration is:

```ini
[Match]
Name=ext

[Network]
DHCP=yes
DNSSEC=no
IPv4Forwarding=yes
IPv6Forwarding=yes
IPv6PrivacyExtensions=true
```

Save these as `20-int.network` and `20-ext.network` under `/etc/systemd/network/`.

Each file has several sections. `[Match]` selects the interface by the name assigned earlier; matching by MAC address is another option. `[Network]` specifies addresses and enables services such as DHCP. Further sections configure those services in more detail.

The IPv6 settings above reproduce the original setup. Working IPv6 service also depends on the upstream network, including prefix delegation; these settings alone do not guarantee IPv6 connectivity.

For other network managers, see the router documentation linked at the end.

##### Masquerading

After configuring the interfaces, the kernel must be allowed to forward packets between them. The `IPv4Forwarding=yes` and `IPv6Forwarding=yes` settings above enable that forwarding.

Forwarding alone does not make private LAN IPv4 addresses reachable from the public Internet. Address ranges such as `10.x.x.x` and `192.168.x.x` are reused in countless separate networks. Internet routers cannot establish a unique public return route to each of them. A reply addressed to a private address would not identify which home network should receive it.

This is why we need masquerading in this setup.

systemd-networkd can configure it automatically using `IPMasquerade=` on the LAN interface. For the IPv4 case discussed here, use `IPMasquerade=ipv4`. This also enables the corresponding forwarding behavior and arranges the necessary packet-filtering rules.

To understand the mechanism, however, let us configure nftables ourselves instead of enabling the automatic option.

Write this configuration:

```text
table inet nat {
  chain postrouting {
    type nat hook postrouting priority 0; policy accept;

    iif int oif ext masquerade
  }
}
```

The rule matches traffic arriving through `int` and leaving through `ext`, then applies masquerading in the postrouting hook. Load the file with `nft -f` followed by its filename.

What does the `masquerade` keyword actually do? Consider a related source-NAT rule:

```text
iif int oif ext snat to <external-interface-ip>
```

Here, `<external-interface-ip>` is a placeholder for an actual address. The rule means: in postrouting, find packets going from the LAN to the Internet and replace their source address with the external interface's address.

The upstream network can then return replies to that external address, and connection tracking lets the router translate those replies back to the appropriate LAN device.

The drawback of explicitly writing a source address is that the rule must know the address in advance and stay in sync if it changes. In the configuration above, only the LAN address is fixed; the external interface obtains its address automatically.

That is where masquerading helps. It performs source NAT while selecting the outgoing interface's address automatically.

If you have read my [article on transparent proxying]({{ '/en/tproxy/' | relative_url }}), some of these ideas may feel familiar. A transparent proxy can also relay selected traffic on behalf of LAN clients, although that does not make an application proxy a general replacement for routing and NAT for every protocol.

#### What comes next?

These settings provide the basic forwarding and address-translation functions of an Arch Linux router.

A router connected to the Internet also needs a firewall. In a home setup, pay particular attention to traffic arriving on the Internet-facing interface. The configuration shown above explains forwarding and NAT; it does not define a complete firewall policy.

I also prefer to keep the router focused on core network services such as NAT, the firewall, and DHCP. Web applications and file-sharing services can run on separate LAN hosts, with destination NAT (DNAT) exposing services where appropriate.

Routing tables are another necessary part of the system, but the interface configuration normally creates the relevant routes automatically, so I have not explored them in depth here.

I hope the process has made the router's role clearer. For further configuration and study, see the ArchWiki pages on routers and network management.

References:

- <https://wiki.archlinux.org/title/Router>
- <https://wiki.archlinux.org/title/Systemd-networkd>
- <https://man.archlinux.org/man/systemd.network.5>
- <https://wiki.nftables.org/wiki-nftables/index.php/Performing_Network_Address_Translation_(NAT)>
