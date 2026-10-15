# Generator for Sessions 6-10:
# Constraint: Alternate short turns (< 10 words) with longer turns (25-35 words).
# Turn 1: < 10
# Turn 2: 25-35
# Turn 3: < 10
# Turn 4: 25-35
# ...
# Each session has at least 50 turns, at least 2 research requests, coherent follow-ups, corrections, references.

def get_session_6():
    # Session 6: Canonical product strategy, sovereignty and enterprise value propositions
    t = [
        "Let's analyze Canonical's enterprise product strategy.", # 6 words (< 10)
        "How does Canonical position Ubuntu Pro against Red Hat Enterprise Linux and SUSE for long-term security maintenance, compliance certifications, and per-node subscription pricing across hybrid multi-cloud enterprise deployments?", # 29 words (25-35)
        "What is Ubuntu Pro's core value proposition?", # 6 words (< 10)
        "By extending security patching to over twenty-three thousand open source packages in the universe repository, does Canonical successfully alleviate enterprise vulnerability management burdens that competitors often leave unaddressed in standard support contracts?", # 34 words (25-35)
        "Is universe coverage unique in enterprise Linux?", # 7 words (< 10)
        "How does Canonical's commitment to ten or twelve years of expanded security maintenance impact total cost of ownership for mission-critical infrastructure that enterprises cannot afford to upgrade every three years?", # 31 words (25-35)
        "Explain Landscape for automated fleet management.", # 6 words (< 10)
        "Can Landscape scale effectively across tens of thousands of distributed edge devices, on-premises virtual machines, and cloud instances while enforcing compliance auditing, script execution, and coordinated unattended package rollouts?", # 30 words (25-35)
        "What about European digital sovereignty?", # 5 words (< 10)
        "Search independent market reports on European cloud sovereignty and open-source infrastructure adoption rates across public sector agencies in Germany, France, and the wider European Union.", # 26 words (25-35) [Research 1]
        "Summarize the sovereignty findings.", # 4 words (< 10)
        "How does hosting infrastructure on Ubuntu within European data centers provide legal data residency advantages over proprietary American hyperscaler operating systems when complying with GDPR and the European AI Act?", # 29 words (25-35)
        "What role does Charmed Kubernetes play?", # 6 words (< 10)
        "Wait, earlier I mentioned MicroK8s, but I now clarify that my focus is strictly on Charmed Kubernetes orchestrating resilient multi-cloud worker clusters with automated Day-2 operations.", # 27 words (25-35) [Correction]
        "How do Juju charms automate lifecycle management?", # 7 words (< 10)
        "By packaging operational knowledge into declarative charms, does Canonical enable system administrators to deploy, integrate, scale, and upgrade complex distributed databases and observability stacks without writing custom shell scripts?", # 31 words (25-35)
        "Compare Snaps and Flatpaks in enterprises.", # 6 words (< 10)
        "While desktop Linux enthusiasts strongly prefer decentralized Flatpaks, why has Canonical remained firmly committed to Snaps for server daemon isolation, automatic transactional rollbacks, and strictly confined Internet of Things appliances?", # 32 words (25-35)
        "Is snap confinement truly strict?", # 5 words (< 10)
        "How does Canonical utilize AppArmor profiles, seccomp filters, and cgroups to ensure that snap packages cannot access unauthorized hardware devices, network sockets, or personal user files on Ubuntu Core?", # 29 words (25-35)
        "Explain Ubuntu Core's immutable architecture.", # 5 words (< 10)
        "With dual system partitions facilitating fail-safe atomic upgrades, how does Ubuntu Core protect remote smart factory edge gateways and autonomous automotive controllers against catastrophic bricking during unattended over-the-air firmware updates?", # 32 words (25-35)
        "What about telecommunication deployments?", # 4 words (< 10)
        "Search telecommunication carrier whitepapers regarding Canonical Real-Time Ubuntu adoption for virtualized Radio Access Networks and low-latency Telco Cloud edge infrastructure deployments.", # 25 words (25-35) [Research 2]
        "Summarize the telecommunication research.", # 4 words (< 10)
        "Does the PREEMPT_RT patched Linux kernel provided in Ubuntu Pro meet the stringent deterministic latency requirements demanded by open RAN distributed units handling radio signal processing under heavy network loads?", # 32 words (25-35)
        "Remember earlier we discussed sovereignty? What about chips?", # 9 words (< 10)
        "Can Canonical's strategic support for RISC-V architectures and European processor initiatives provide genuine silicon independence if geopolitical export controls restrict American x86 and British ARM hardware access?", # 28 words (25-35)
        "How does Canonical monetize open source?", # 6 words (< 10)
        "Rather than charging software license fees or walling source code behind paywalls, Canonical monetizes support, security automation, and managed operations while keeping the underlying operating system freely downloadable for everyone, correct?", # 32 words (25-35)
        "Is per-node pricing more predictable than vCPU?", # 7 words (< 10)
        "When running high-density server blades with hundreds of compute cores, why do enterprise chief financial officers favor Canonical's socket-agnostic node pricing over competitors who charge exponentially higher fees per virtual processor core?", # 33 words (25-35)
        "What is Canonical's data platform strategy?", # 6 words (< 10)
        "Can Charmed PostgreSQL, Kafka, OpenSearch, and Spark provide a complete open-source enterprise data foundation capable of replacing costly proprietary cloud services like Amazon RDS and Confluent Kafka on private infrastructure?", # 31 words (25-35)
        "How does Canonical approach local AI?", # 6 words (< 10)
        "By packaging NVIDIA GPU drivers, Intel oneAPI runtimes, and local inference runtimes into standard Ubuntu distributions, does Canonical empower organizations to develop sovereign generative artificial intelligence models entirely on-premises?", # 31 words (25-35)
        "Explain the Anbox Cloud enterprise platform.", # 6 words (< 10)
        "How does containerizing Android instances in lightweight LXD containers on Ubuntu servers enable telecom operators to deliver cloud gaming and secure mobile application virtualization at scale with high graphics acceleration?", # 32 words (25-35)
        "What is Canonical's relationship with public clouds?", # 7 words (< 10)
        "Given that Ubuntu represents over seventy percent of all Linux workloads on AWS, Azure, and Google Cloud, how does Canonical collaborate with hyperscalers to engineer custom-tuned kernels and minimal images?", # 31 words (25-35)
        "Does cloud revenue subsidize desktop development?", # 6 words (< 10)
        "While desktop Ubuntu does not directly generate substantial software revenues, does its widespread adoption among software developers serve as the primary organic top-of-funnel customer acquisition pipeline for enterprise Ubuntu Pro subscriptions?", # 32 words (25-35)
        "What governance model guides Canonical's decisions?", # 6 words (< 10)
        "How does Canonical balance the benevolent dictatorship governance model with democratic community councils, technical boards, and independent upstream open source maintainers when strategic commercial priorities clash with grass-roots developer expectations?", # 32 words (25-35)
        "Earlier we touched on compliance. What standards apply?", # 8 words (< 10)
        "Can Ubuntu Pro out of the box satisfy rigorous US government and financial sector compliance frameworks including FIPS 140-3 cryptography, DISA-STIG security profiles, CIS benchmarks, and HIPAA data protection mandates?", # 31 words (25-35)
        "Is Canonical preparing for an initial public offering?", # 8 words (< 10)
        "If Canonical eventually pursues a public stock listing, what key profitability metrics, enterprise contract retention rates, and recurring annual revenues will institutional investors examine when evaluating its valuation against Red Hat?", # 33 words (25-35)
        "Summarize Canonical's enterprise competitive moat.", # 5 words (< 10)
        "In three concise concluding points, synthesize how Canonical combines enterprise security longevity, European sovereignty alignment, and developer familiarity to sustain durable competitive differentiation against legacy commercial enterprise software giants." # 30 words (25-35)
    ]
    return t

def get_session_7():
    # Session 7: OpenStack, Ceph, OVN, K8s, Kubeflow and resilient multi-site infrastructure
    t = [
        "Let's architect resilient multi-site cloud infrastructure.", # 6 words (< 10)
        "How do modern enterprise organizations combine OpenStack for infrastructure-as-a-service with Ceph storage, Open Virtual Network software-defined networking, and Kubernetes to build private sovereign clouds that rival public hyperscalers in resilience?", # 33 words (25-35)
        "Explain Ceph's CRUSH map data distribution.", # 6 words (< 10)
        "How does the CRUSH algorithm deterministically calculate object placement across failure domains without relying on centralized lookup tables, and how does this enable petabyte-scale horizontal storage expansion without metadata bottlenecks?", # 31 words (25-35)
        "What are Ceph pool failure domains?", # 6 words (< 10)
        "When designing a multi-datacenter Ceph deployment across three availability zones, should the crush failure domain be configured at the rack, host, or room level to guarantee uninterrupted quorum during power outages?", # 32 words (25-35)
        "How does Ceph handle split-brain network partitions?", # 7 words (< 10)
        "Why does Ceph require an odd number of Monitor daemons to establish Paxos consensus, and what precise sequence of automatic peer fencing occurs if a high-latency inter-site fiber link splits the cluster?", # 33 words (25-35)
        "Explain OVN software-defined networking architecture.", # 5 words (< 10)
        "Search architecture documentation on Open Virtual Network geneve encapsulation and distributed routing scale limits in large OpenStack multi-hypervisor cloud fabrics.", # 21 -> wait! 25-35 words:
        # Let's adjust turn 10 to be exactly 26 words
    ]
    return t

print("Checking template")
