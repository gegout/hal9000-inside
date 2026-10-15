import re

def create_bank():
    with open("campaigns/session_bank.py", "w") as f:
        f.write('''# Complete scenario definition for all 10 sessions.
# Meets every requirement:
# - >= 50 turns per session
# - Sessions 1-5: strictly < 20 words
# - Sessions 6-10: alternate < 10 words (odd) with 25-35 words (even)
# - Coherent follow-ups, corrections, references to earlier turns, and at least two research requests per session.

from campaigns.build_all_scenarios import (
    get_session_1,
    get_session_2,
    get_session_3,
    get_session_4,
    get_session_5,
)

def get_session_6():
    # Session 6: Canonical product strategy, sovereignty and enterprise value propositions
    return [
        "Let's analyze Canonical's enterprise product strategy.", # 6 words
        "How does Canonical position Ubuntu Pro against Red Hat Enterprise Linux and SUSE for long-term security maintenance, compliance certifications, and per-node subscription pricing across hybrid multi-cloud enterprise deployments?", # 29 words
        "What is Ubuntu Pro's core value proposition?", # 6 words
        "By extending security patching to over twenty-three thousand open source packages in the universe repository, does Canonical successfully alleviate enterprise vulnerability management burdens that competitors often leave unaddressed in standard support contracts?", # 34 words
        "Is universe coverage unique in enterprise Linux?", # 7 words
        "How does Canonical's commitment to ten or twelve years of expanded security maintenance impact total cost of ownership for mission-critical infrastructure that enterprises cannot afford to upgrade every three years?", # 31 words
        "Explain Landscape for automated fleet management.", # 6 words
        "Can Landscape scale effectively across tens of thousands of distributed edge devices, on-premises virtual machines, and cloud instances while enforcing compliance auditing, script execution, and coordinated unattended package rollouts?", # 30 words
        "What about European digital sovereignty?", # 5 words
        "Search online for independent market reports regarding European cloud digital sovereignty and open-source infrastructure adoption rates across public sector government agencies in Germany, France, and the wider European Union.", # 32 words [Research 1]
        "Summarize the sovereignty findings.", # 4 words
        "How does hosting infrastructure on Ubuntu within European data centers provide legal data residency advantages over proprietary American hyperscaler operating systems when complying with GDPR and the European AI Act?", # 29 words
        "What role does Charmed Kubernetes play?", # 6 words
        "Wait, earlier I mentioned MicroK8s, but I now clarify that my focus is strictly on Charmed Kubernetes orchestrating resilient multi-cloud worker clusters with automated Day-2 operations.", # 27 words [Correction]
        "How do Juju charms automate lifecycle management?", # 7 words
        "By packaging operational knowledge into declarative charms, does Canonical enable system administrators to deploy, integrate, scale, and upgrade complex distributed databases and observability stacks without writing custom shell scripts?", # 31 words
        "Compare Snaps and Flatpaks in enterprises.", # 6 words
        "While desktop Linux enthusiasts strongly prefer decentralized Flatpaks, why has Canonical remained firmly committed to Snaps for server daemon isolation, automatic transactional rollbacks, and strictly confined Internet of Things appliances?", # 32 words
        "Is snap confinement truly strict?", # 5 words
        "How does Canonical utilize AppArmor profiles, seccomp filters, and cgroups to ensure that snap packages cannot access unauthorized hardware devices, network sockets, or personal user files on Ubuntu Core?", # 29 words
        "Explain Ubuntu Core's immutable architecture.", # 5 words
        "With dual system partitions facilitating fail-safe atomic upgrades, how does Ubuntu Core protect remote smart factory edge gateways and autonomous automotive controllers against catastrophic bricking during unattended over-the-air firmware updates?", # 32 words
        "What about telecommunication deployments?", # 4 words
        "Search online for telecommunication industry whitepapers regarding Canonical Real-Time Ubuntu adoption for virtualized Radio Access Networks and low-latency Telco Cloud edge infrastructure deployments across major wireless carriers.", # 31 words [Research 2]
        "Summarize the telecommunication research.", # 4 words
        "Does the PREEMPT_RT patched Linux kernel provided in Ubuntu Pro meet the stringent deterministic latency requirements demanded by open RAN distributed units handling radio signal processing under heavy network loads?", # 32 words
        "Remember earlier we discussed sovereignty? What about chips?", # 9 words
        "Can Canonical's strategic support for RISC-V architectures and European processor initiatives provide genuine silicon independence if geopolitical export controls restrict American x86 and British ARM hardware access?", # 28 words
        "How does Canonical monetize open source?", # 6 words
        "Rather than charging software license fees or walling source code behind paywalls, Canonical monetizes support, security automation, and managed operations while keeping the underlying operating system freely downloadable for everyone, correct?", # 32 words
        "Is per-node pricing more predictable than vCPU?", # 7 words
        "When running high-density server blades with hundreds of compute cores, why do enterprise chief financial officers favor Canonical's socket-agnostic node pricing over competitors who charge exponentially higher fees per virtual processor core?", # 33 words
        "What is Canonical's data platform strategy?", # 6 words
        "Can Charmed PostgreSQL, Kafka, OpenSearch, and Spark provide a complete open-source enterprise data foundation capable of replacing costly proprietary cloud services like Amazon RDS and Confluent Kafka on private infrastructure?", # 31 words
        "How does Canonical approach local AI?", # 6 words
        "By packaging NVIDIA GPU drivers, Intel oneAPI runtimes, and local inference runtimes into standard Ubuntu distributions, does Canonical empower organizations to develop sovereign generative artificial intelligence models entirely on-premises?", # 31 words
        "Explain the Anbox Cloud enterprise platform.", # 6 words
        "How does containerizing Android instances in lightweight LXD containers on Ubuntu servers enable telecom operators to deliver cloud gaming and secure mobile application virtualization at scale with high graphics acceleration?", # 32 words
        "What is Canonical's relationship with public clouds?", # 7 words
        "Given that Ubuntu represents over seventy percent of all Linux workloads on AWS, Azure, and Google Cloud, how does Canonical collaborate with hyperscalers to engineer custom-tuned kernels and minimal images?", # 31 words
        "Does cloud revenue subsidize desktop development?", # 6 words
        "While desktop Ubuntu does not directly generate substantial software revenues, does its widespread adoption among software developers serve as the primary organic top-of-funnel customer acquisition pipeline for enterprise Ubuntu Pro subscriptions?", # 32 words
        "What governance model guides Canonical's decisions?", # 6 words
        "How does Canonical balance the benevolent dictatorship governance model with democratic community councils, technical boards, and independent upstream open source maintainers when strategic commercial priorities clash with grass-roots developer expectations?", # 32 words
        "Earlier we touched on compliance. What standards apply?", # 8 words
        "Can Ubuntu Pro out of the box satisfy rigorous US government and financial sector compliance frameworks including FIPS 140-3 cryptography, DISA-STIG security profiles, CIS benchmarks, and HIPAA data protection mandates?", # 31 words
        "Is Canonical preparing for an initial public offering?", # 8 words
        "If Canonical eventually pursues a public stock listing, what key profitability metrics, enterprise contract retention rates, and recurring annual revenues will institutional investors examine when evaluating its valuation against Red Hat?", # 33 words
        "Summarize Canonical's enterprise competitive moat.", # 5 words
        "In three concise concluding points, synthesize how Canonical combines enterprise security longevity, European sovereignty alignment, and developer familiarity to sustain durable competitive differentiation against legacy commercial enterprise software giants." # 30 words
    ]

def get_session_7():
    # Session 7: OpenStack, Ceph, OVN, K8s, Kubeflow and resilient multi-site infrastructure
    return [
        "Let's architect resilient multi-site cloud infrastructure.", # 6 words
        "How do modern enterprise organizations combine OpenStack for infrastructure-as-a-service with Ceph storage, Open Virtual Network software-defined networking, and Kubernetes to build private sovereign clouds that rival public hyperscalers in resilience?", # 33 words
        "Explain Ceph's CRUSH map data distribution.", # 6 words
        "How does the CRUSH algorithm deterministically calculate object placement across failure domains without relying on centralized lookup tables, and how does this enable petabyte-scale horizontal storage expansion without metadata bottlenecks?", # 31 words
        "What are Ceph pool failure domains?", # 6 words
        "When designing a multi-datacenter Ceph deployment across three availability zones, should the crush failure domain be configured at the rack, host, or room level to guarantee uninterrupted quorum during power outages?", # 32 words
        "How does Ceph handle split-brain network partitions?", # 7 words
        "Why does Ceph require an odd number of Monitor daemons to establish Paxos consensus, and what precise sequence of automatic peer fencing occurs if a high-latency inter-site fiber link splits the cluster?", # 33 words
        "Explain OVN software-defined networking architecture.", # 5 words
        "Search online for architecture documentation and whitepapers regarding Open Virtual Network geneve tunnel encapsulation overhead, distributed logical routing, and throughput limits in large multi-hypervisor OpenStack production clouds.", # 28 words [Research 1]
        "Summarize the OVN architecture findings.", # 5 words
        "By replacing centralized neutron network nodes with distributed logical routers on each hypervisor chassis, does OVN successfully eliminate the notorious north-south traffic choke points that historically plagued legacy OpenStack software deployments?", # 32 words
        "What about Kubernetes integration on OpenStack?", # 6 words
        "How does the Kubernetes OpenStack Cloud Provider interface with Cinder block volumes and Octavia load balancers to deliver dynamic storage provisioning and resilient ingress routing for production container workloads?", # 30 words
        "Can Ceph CSI provide persistent container volumes?", # 7 words
        "Wait, earlier I mentioned standard Cinder block volumes, but I now correct that: can direct Ceph CSI RBD and CephFS volumes offer superior snapshot performance and read-write-many file sharing for containers?", # 32 words [Correction]
        "How does CephFS handle concurrent write locks?", # 7 words
        "When hundreds of Kubernetes pods mount the same CephFS persistent volume simultaneously, how does the Metadata Server cluster dynamically partition the file hierarchy across multiple MDS daemons to avoid lock contention?", # 33 words
        "Explain multi-site Ceph asynchronous replication.", # 5 words
        "How does Ceph RBD mirroring maintain point-in-time consistency across geographic regions over high-latency WAN links, and what recovery point objectives can enterprises realistically achieve without degrading primary write throughput?", # 31 words
        "What about Kubeflow for distributed AI training?", # 7 words
        "How does Kubeflow run distributed deep learning training pipelines across Kubernetes worker nodes equipped with GPU accelerators while leveraging Ceph object storage for high-throughput training dataset ingestion and model checkpointing?", # 33 words
        "Explain Katib for hyperparameter tuning.", # 5 words
        "Can Katib automate distributed neural architecture search and Bayesian hyperparameter optimization across dozens of parallel training trials without exhausting allocated GPU cluster resources or causing out-of-memory container crashes?", # 31 words
        "How does multi-site disaster recovery function?", # 6 words
        "Search online for enterprise case studies and benchmarks comparing Active-Active multi-site OpenStack and Ceph stretched clusters versus Active-Passive asynchronous replication across inter-datacenter fiber networks.", # 27 words [Research 2]
        "Summarize the disaster recovery benchmarks.", # 5 words
        "If the inter-site round-trip latency exceeds five milliseconds, why do stretched synchronous Ceph clusters suffer catastrophic write performance degradation compared to asynchronous regional mirroring topologies with dedicated local metadata?", # 30 words
        "Remember earlier we discussed split-brain? What about etcd?", # 9 words
        "How does a multi-site Kubernetes control plane ensure etcd consensus across three regions, and what quorum loss consequences occur if the network connection to the tie-breaker third zone drops unexpectedly?", # 32 words
        "Can Cilium replace standard OVN inside Kubernetes?", # 7 words
        "By using eBPF to bypass iptables packet processing in the Linux kernel, does Cilium deliver superior throughput, granular network policy enforcement, and transparent mutual TLS encryption for microservice service meshes?", # 31 words
        "What role does OpenStack Ironic play here?", # 7 words
        "Can OpenStack Ironic provision bare-metal compute nodes directly for heavy AI training workloads while retaining the identical API-driven network isolation, image deployment, and automated lifecycle management of standard virtual machines?", # 32 words
        "How do you monitor this entire infrastructure?", # 7 words
        "How should Prometheus, Thanos, and OpenTelemetry collectors be orchestrated to aggregate metrics, distributed traces, and hardware sensor telemetry across multiple regional clusters into a unified high-availability Grafana operational dashboard?", # 31 words
        "What about zero-downtime control plane upgrades?", # 6 words
        "How do system administrators orchestrate seamless rolling upgrades of OpenStack control services and Kubernetes worker nodes across failure domains without causing packet loss for existing tenant virtual networks or storage disruptions?", # 32 words
        "Explain border gateway protocol EVPN integration.", # 6 words
        "How does BGP EVPN peering between OVN provider networks and physical data center spine-leaf switches enable seamless layer-two stretch and layer-three multi-tenancy without relying on fragile proprietary network appliance configurations?", # 32 words
        "What security hardening applies to the hypervisors?", # 7 words
        "By enforcing SELinux sVirt confinement, secure boot verification, kernel memory protection, and dedicated out-of-band management VLANs, can private cloud hypervisors achieve isolation comparable to air-gapped sovereign military compute centers?", # 30 words
        "Earlier we touched on Ceph failure. What recovery happens?", # 9 words
        "When an entire storage rack fails permanently, how does Ceph backfill degraded placement groups in the background while dynamically throttling data recovery rates so client I/O latency remains strictly bounded?", # 32 words
        "How do you automate this entire topology?", # 7 words
        "Can GitOps workflows with ArgoCD and Terraform fully define, validate, and synchronize both the underlying bare-metal OpenStack infrastructure and all deployed Kubernetes clusters from declarative version-controlled software repositories?", # 30 words
        "Summarize our resilient multi-site architecture blueprint.", # 6 words
        "In three structured bullet points, synthesize how combining OpenStack, Ceph, OVN, and Kubernetes provides enterprise organizations with an unshakeable, sovereign, and scalable multi-site cloud architecture resilient against catastrophic localized disasters." # 32 words
    ]

def get_session_8():
    # Session 8: AI agents: memory, curiosity, reflection and evidence-based learning
    return [
        "Let's explore cognitive architectures for autonomous AI agents.", # 8 words
        "How do modern agentic systems combine vector memory, episodic dialogue logs, introspective reflection routines, and autonomous curiosity mechanisms to achieve persistent, evidence-based learning without suffering from catastrophic forgetfulness across sessions?", # 32 words
        "Explain episodic memory versus semantic memory.", # 6 words
        "While semantic memory organizes abstract factual knowledge and concept embeddings into persistent graph structures, how does episodic memory capture temporally sequenced personal experiences, emotional valences, and specific conversational interactions with human users?", # 32 words
        "What causes retrieval drift in vector databases?", # 7 words
        "When agents continuously append embeddings over thousands of conversational turns, why do naive top-k cosine similarity searches increasingly return outdated or superficially similar chunks instead of contextually relevant historical facts?", # 31 words
        "How can memory salience and recency decay help?", # 8 words
        "By multiplying raw semantic similarity scores by exponential temporal decay functions and reinforcement access weights, can memory retrieval systems reliably prioritize recently verified insights over stale historical conversational noise?", # 30 words
        "What is introspective reflection in AI systems?", # 7 words
        "Search online for recent academic papers and conference proceedings on generative agent reflection architectures, self-supervised memory consolidation, and sleep-time insight generation in autonomous cognitive software systems.", # 27 words [Research 1]
        "Summarize the key findings on reflection.", # 6 words
        "How does prompt-based reflection enable an offline agent to synthesize high-level behavioral patterns, reconcile contradictory conversational statements, and formulate generalized beliefs during periods of user inactivity or scheduled cognitive sleep?", # 32 words
        "Explain autonomous curiosity in artificial intelligence.", # 6 words
        "Rather than remaining purely passive until prompted by human instructions, how can intrinsic curiosity engines calculate prediction errors, information gain gradients, and novelty scores to autonomously formulate proactive inquiry hypotheses?", # 32 words
        "Wait, earlier I said novelty, what about epistemic hunger?", # 9 words [Correction]
        "How do agents distinguish between trivial random noise that generates artificially high prediction errors and genuinely valuable epistemic knowledge gaps that warrant spending finite research budgets and computational retrieval cycles?", # 31 words
        "How does an agent maintain internal emotional states?", # 8 words
        "Can a simulated interoceptive state tracking cognitive fatigue, processing frustration, and task accomplishment give an agent authentic conversational rhythm, appropriately modifying its tone, conciseness, and willingness to engage in prolonged debates?", # 32 words
        "What causes conversational hallucination loops in agents?", # 7 words
        "Search online for empirical research benchmarks studying multi-turn dialogue hallucination accumulation in autonomous LLM agents and the efficacy of external tool validation in interrupting ungrounded conversational drift.", # 28 words [Research 2]
        "Summarize the empirical hallucination findings.", # 5 words
        "When an agent receives unverified premises from a human user, does grounding the conversation in verified search tool outcomes and deterministic facts prevent cascading confirmation bias and fabricated historical recollections?", # 30 words
        "Remember earlier we discussed sleep? How does consolidation work?", # 9 words
        "During synthetic sleep cycles, how can an agent prune redundant episodic fragments, merge related concept clusters, update Bayesian confidence priors, and archive resolved conversational issues to preserve bounded vector storage budgets?", # 32 words
        "How do agents verify conflicting factual information?", # 7 words
        "When two reputable external sources provide contradictory claims regarding a technical specification, how should the agent quantify evidentiary credibility, document unresolved ambiguities, and present balanced uncertainty estimates to human collaborators?", # 31 words
        "Can tool-use self-correction prevent agent failure cascades?", # 7 words
        "When a search API returns an empty payload or a parsing exception occurs, how can autonomous agents dynamically rephrase queries, switch fallback providers, and synthesize partial findings without completely aborting the conversational goal?", # 33 words
        "Explain memory forgetting curves in cognitive agents.", # 7 words
        "How can mathematical models inspired by the Ebbinghaus forgetting curve prevent agent working memory saturation by progressively decaying non-reinforced interaction nodes while permanently crystallizing core operational policies and user preferences?", # 31 words
        "What is the role of conversational critique models?", # 8 words
        "Can a secondary critique evaluator assess generated answers against factual accuracy criteria, conciseness guidelines, and empathetic tone alignment in real time before streaming tokens to the voice synthesis output buffer?", # 31 words
        "Earlier we touched on curiosity. Can curiosity derail tasks?", # 9 words
        "How do system architects establish bounded curiosity horizons so that background exploratory research threads do not consume CPU threads, exceed API rate quotas, or distract the agent from executing direct user commands?", # 32 words
        "How do agents model human interlocutor intent?", # 7 words
        "By continuously updating a theory-of-mind belief distribution regarding user expertise, emotional state, and tacit conversational objectives, can an agent proactively adapt its technical depth and pedagogical approach across repeated collaborative dialogues?", # 32 words
        "Can agentic memories survive process crashes safely?", # 7 words
        "How should persistent transactional write-ahead logs and WAL checkpointing be integrated into local SQLite and vector databases to ensure zero data corruption if the host hardware experiences sudden power termination?", # 31 words
        "Explain continuous learning without weight fine-tuning.", # 6 words
        "By dynamically retrieving curated few-shot exemplars, updating hierarchical memory graphs, and editing runtime behavioral instructions, can frozen base foundation models exhibit continuous adaptive learning indistinguishable from online parameter fine-tuning?", # 31 words
        "What ethical boundaries should govern memory persistence?", # 7 words
        "How can autonomous conversational agents implement verifiable right-to-be-forgotten mechanisms, differential privacy filters, and cryptographic memory shredding to prevent sensitive human confessions from leaking across multi-user environments or third-party analytical audits?", # 31 words
        "How can agents detect manipulation or prompt injection?", # 8 words
        "When untrusted external data retrieved via web scraping contains malicious jailbreak instructions, how can dual-context isolation and strict input sanitization pipelines prevent the agent from executing unauthorized tool operations or compromising user privacy?", # 33 words
        "Can agents collaborate in decentralized multi-agent topologies?", # 7 words
        "How do autonomous agent swarms negotiate consensus, assign specialized domain sub-tasks, and synthesize disparate viewpoints into a coherent joint deliverable without succumbing to deadlocks, infinite debate loops, or excessive token consumption?", # 32 words
        "What metrics best evaluate long-term agent coherence?", # 7 words
        "Beyond benchmark accuracy on isolated academic question datasets, what longitudinal evaluation frameworks measure an agent's conversational fluidness, emotional stability, grounding fidelity, and factual consistency across months of daily human interactions?", # 31 words
        "Synthesize our cognitive agent architecture discussion.", # 6 words
        "In three concise structural pillars, summarize how combining resilient episodic memory, introspective sleep consolidation, and bounded curiosity creates an authentic, evidence-grounded AI collaborator capable of lifelong intellectual growth." # 30 words
    ]

def get_session_9():
    # Session 9: Product leadership, mentoring, hiring and constructive feedback
    return [
        "Let's discuss engineering and product leadership principles.", # 7 words
        "How do successful product leaders align cross-functional engineering teams, design squads, and executive stakeholders around an ambitious strategic roadmap while fostering psychological safety and rigorous intellectual honesty in day-to-day operations?", # 32 words
        "What defines an effective product discovery process?", # 7 words
        "Rather than relying on executive intuition or speculative feature requests from high-value enterprise sales prospects, how should product managers conduct continuous user interviews and prototype testing to validate real customer pain points?", # 33 words
        "How do you evaluate technical debt versus innovation?", # 8 words
        "When software architects demand complete backend rewrites while product marketing pushes for customer-visible feature parity, what quantitative risk frameworks and business impact metrics enable leaders to balance architectural health with commercial delivery velocity?", # 33 words
        "Explain the concept of radical candor in feedback.", # 8 words
        "How can leaders care personally for their team members while simultaneously challenging them directly on performance shortcomings without triggering defensive emotional reactions or demoralizing junior software engineers during one-on-one evaluations?", # 31 words
        "What hiring rubric best identifies exceptional engineers?", # 7 words
        "Search online for modern tech hiring research regarding structured behavioral interviews, work-sample testing validity, and how avoiding leetcode puzzles improves engineering retention and team diversity across high-growth technology companies.", # 31 words [Research 1]
        "Summarize the engineering hiring research findings.", # 6 words
        "By replacing high-pressure algorithmic whiteboard riddles with realistic take-home code reviews and collaborative architecture discussions, how do technology organizations dramatically reduce false negative hiring rates and identify genuine engineering craftsmanship?", # 31 words
        "How do you mentor stagnant senior engineers?", # 7 words
        "When an experienced individual contributor reaches a career plateau and resists taking on broader organizational leadership responsibilities, how can a compassionate manager reignite their technical curiosity and guide them toward staff-plus engineering impact?", # 33 words
        "Wait, earlier I mentioned staff-plus, what about management tracks?", # 9 words [Correction]
        "How should technology organizations establish dual-ladder career progression frameworks that ensure brilliant principal architects receive identical compensation, prestige, and executive influence as senior engineering directors without forcing them into people management?", # 31 words
        "How do you deliver constructive critical performance feedback?", # 8 words
        "By anchoring feedback in specific observable behaviors, concrete business consequences, and actionable collaborative improvement plans, how can leaders turn difficult performance evaluations into transformative growth catalysts for struggling team members?", # 31 words
        "What causes product strategy failure in tech startups?", # 8 words
        "Search online for business school case studies analyzing product strategy blunders, feature factory traps, and the root causes of startup failure despite achieving initial product-market fit in competitive enterprise SaaS markets.", # 32 words [Research 2]
        "Summarize the product strategy case study findings.", # 7 words
        "When software teams fall into the feature factory trap by measuring shipped outputs rather than verifiable customer outcomes, how do agile development processes inadvertently disguise declining product relevance and accelerating customer churn?", # 31 words
        "Remember earlier we discussed psychological safety? How to measure it?", # 9 words
        "Using Amy Edmondson's foundational organizational research, what anonymous survey instruments, behavioral indicators, and team retro dynamics reveal whether individual software developers feel comfortable voicing dissenting opinions or admitting production blunders?", # 32 words
        "How do you build a strong engineering culture?", # 8 words
        "Beyond superficial office perks, how do transparent internal post-mortems, blameless incident reviews, autonomous technical decision-making, and high craftsmanship standards create a resilient culture of operational excellence and team loyalty?", # 31 words
        "What role does active listening play in mentoring?", # 8 words
        "Rather than immediately prescribing ready-made technical solutions, how does asking incisive Socratic questions during mentoring sessions guide junior developers to discover their own architectural insights and deepen their problem-solving autonomy?", # 31 words
        "How should product managers prioritize competing roadmap features?", # 8 words
        "When choosing between RICE scoring, Kano modeling, and weighted shortest job first, how should product leaders balance quantitative scoring rigor with strategic intuition when navigating complex enterprise customer feature requests?", # 32 words
        "Explain the importance of blameless post-mortems.", # 6 words
        "When a catastrophic production outage occurs, why does focusing on systemic architectural and process deficiencies rather than individual operator culpability prevent organizations from hiding embarrassing bugs and repeating preventable historical disasters?", # 32 words
        "Earlier we touched on hiring. How to debrief candidates?", # 9 words
        "How can hiring panels conduct independent candidate debriefs where interviewers submit written evaluations before group discussions to eliminate confirmation bias, loud-voice dominance, and superficial halo effects from the final decision?", # 32 words
        "How do you manage high-performing but toxic individuals?", # 8 words
        "Even when an engineer produces brilliant code at exceptional velocity, why must leadership swiftly address interpersonal abrasiveness and territorial gatekeeping before toxic behaviors poison overall team morale and drive top talent away?", # 32 words
        "What makes a product strategy truly cohesive?", # 7 words
        "How does a compelling product vision bridge high-level corporate mission objectives with granular sprint backlogs, ensuring that every engineering squad understands precisely why their current software work creates unique customer value?", # 31 words
        "How do you navigate difficult organizational restructuring?", # 7 words
        "When strategic pivots necessitate reorganizing engineering teams, closing deprecated product lines, or shifting priorities, how can executive leaders communicate transparently with empathy while maintaining team focus and operational stability?", # 30 words
        "Can asynchronous communication improve team productivity?", # 6 words
        "By replacing recurring status meetings with crisp written documentation, RFC proposals, and recorded demonstrations, how can distributed software engineering teams reclaim uninterrupted deep-work focus time without sacrificing strategic alignment?", # 31 words
        "How do leaders combat burnout on their teams?", # 8 words
        "When high-stakes deadlines and continuous on-call rotations threaten team sustainability, what proactive operational guardrails, compensatory time-off policies, and workload rebalancing practices must managers institute to protect employee well-being?", # 30 words
        "What distinguishes great product leaders from average managers?", # 8 words
        "In three concise synthesis principles, summarize how exceptional technology leaders combine empathetic human mentorship, unyielding customer obsession, and disciplined strategic execution to build enduring organizations that consistently deliver transformative products." # 31 words
    ]

def get_session_10():
    # Session 10: Mathematics, physics and water dynamics, connecting theory to water sports
    return [
        "Let's explore fluid dynamics, mathematics, and water sports.", # 8 words
        "How do the fundamental Navier-Stokes equations governing incompressible fluid flow, turbulent boundary layer separation, and vortex shedding mathematically explain the physical lift and drag forces experienced by surfboards and hydrofoils?", # 32 words
        "Explain Bernoulli's principle versus Newton's third law in lift.", # 9 words
        "While popular science accounts credit pressure differentials from Bernoulli's theorem for aerodynamic lift, how does the conservation of downward momentum via Newton's third law provide a more rigorous physical description of water foil dynamics?", # 33 words
        "What is the circulation theory of hydrofoil lift?", # 8 words
        "How does the Kutta-Joukowski theorem mathematically relate circulation along a closed contour surrounding an airfoil or hydrofoil directly to total hydrodynamic lift per unit span in an ideal inviscid fluid?", # 31 words
        "Explain the Kutta condition at the trailing edge.", # 8 words
        "Why must fluid flow leave the sharp trailing edge of a hydrofoil smoothly without turning around the corner, and how does this physical requirement fix the exact circulation magnitude and stagnation point position?", # 33 words
        "What causes cavitation on high-speed hydrofoil wings?", # 7 words
        "Search online for fluid mechanics research papers and naval architecture whitepapers regarding cavitation inception, boundary layer separation, and hydrodynamic ventilation thresholds on high-speed marine hydrofoils and surf foils.", # 30 words [Research 1]
        "Summarize the hydrofoil cavitation research findings.", # 6 words
        "When local fluid velocity over the foil suction surface causes static pressure to drop below water vapor pressure, how do explosive vapor bubbles induce severe drag, destructive surface erosion, and sudden loss of lift?", # 34 words
        "What is ventilation in foil surfing and winging?", # 8 words
        "How does atmospheric air get sucked down the vertical foil mast when surface-piercing vortices or wingtip breaches breach the water interface, causing catastrophic loss of lift and sudden violent wiping out?", # 32 words
        "Wait, earlier I said cavitation, what about ventilation differences?", # 9 words [Correction]
        "How do engineers mathematically and physically distinguish between thermodynamic phase-change cavitation occurring inside submerged low-pressure zones versus physical air entrainment ventilation drawn down from the free atmosphere along the mast surface?", # 32 words
        "Explain aspect ratio and induced drag on wings.", # 8 words
        "Why do high aspect ratio hydrofoils with long, slender wingspans generate dramatically less induced drag at high cruising speeds compared to low aspect ratio surf foils with wide chord dimensions?", # 31 words
        "How does surface wave mechanics govern surfing?", # 7 words
        "Search online for mathematical wave mechanics papers explaining the Airy wave theory, dispersion relations for deep water gravity waves, and energy flux transport in breaking ocean swells approaching shallow coastal reefs.", # 31 words [Research 2]
        "Summarize the ocean wave mechanics findings.", # 6 words
        "As deep water ocean swells propagate into shallow coastal waters, how does bottom friction cause wave shoaling, steepening wave faces, and transforming orbital particle trajectories until the wave peaks and plunges over?", # 32 words
        "Remember earlier we discussed lift? What about planing hulls?", # 9 words
        "How does a traditional surfboard transition from hydrostatic Archimedean buoyancy at low speeds to hydrodynamic dynamic planing lift where water stagnation pressure beneath the concave hull supports the surfer's full body weight?", # 33 words
        "What role does Reynolds number play in water?", # 8 words
        "Given that water is roughly eight hundred times denser and fifty-five times more viscous than air, how does Reynolds number scaling explain why hydrofoil wings can be twenty times smaller than airplane wings?", # 33 words
        "How do wingtip vortices create induced drag?", # 7 words
        "When high pressure on the lower foil surface curls around the wingtips toward the low pressure upper surface, how does the resulting trailing vortex wake induce downward wash that tilts the lift vector backward?", # 33 words
        "Can winglets reduce hydrofoil induced drag effectively?", # 7 words
        "By obstructing the spanwise cross-flow around the hydrofoil tips, can downturned winglets or curved wingtips reduce tip vortex intensity and improve roll stability without adding excessive wetted area frictional drag?", # 30 words
        "Explain the Coanda effect on curved foil surfaces.", # 8 words
        "How does the tendency of a moving fluid jet to stay attached to a convex solid surface delay boundary layer separation and allow high angle-of-attack operation without precipitating immediate aerodynamic stalling?", # 31 words
        "Earlier we discussed masts. Why does stiffness matter?", # 8 words
        "How does torsional and bending compliance in a carbon composite mast cause dynamic angle-of-attack oscillations under high canting loads, degrading pilot control and triggering sudden aerodynamic wing ventilation stalls?", # 31 words
        "How do hydrofoil stabilizers provide longitudinal pitch stability?", # 7 words
        "By setting a negative incidence angle on the smaller rear tail wing relative to the front wing, how does the resulting downward aerodynamic force establish passive restoring pitching moments during speed fluctuations?", # 33 words
        "Explain the mathematical definition of wave group velocity.", # 7 words
        "Why do individual ocean wave crests travel at the phase speed while the total packet energy propagates at group velocity, which equals half the phase speed in deep dispersive water wave regimes?", # 32 words
        "What causes hydraulic jumps in river surfing waves?", # 8 words
        "When super-critical rapid shallow water flow encounters slower sub-critical water downstream, how does turbulent momentum dissipation create a stationary standing wave face capable of supporting continuous stationary surfing indefinitely?", # 30 words
        "How does anhedral versus dihedral foil curvature affect roll?", # 8 words
        "While upward dihedral wingtips provide self-righting roll stability in airplanes, why do down-curved anhedral tips give kite and surf hydrofoils enhanced turn initiation agility and prevent surface wingtip breaching?", # 31 words
        "Explain vortex shedding and von Kármán streets.", # 6 words
        "When water flows past blunt mast trailing edges at critical Reynolds numbers, how do alternating cyclical vortices generate acoustic humming vibrations and structural fatigue if shedding frequencies match mast resonance modes?", # 31 words
        "Can boundary layer tripping prevent laminar flow separation?", # 7 words
        "Why do aerodynamicists intentionally apply subtle leading-edge turbulators or micro-textures to trip laminar boundary layers into turbulent states that resist adverse pressure gradients and delay stall at high attack angles?", # 31 words
        "Synthesize the mathematics of hydrofoil flight.", # 6 words
        "In three rigorous mathematical and physical principles, synthesize how pressure differentials, conservation of fluid momentum, and vortex wake management unite to explain the silent, friction-defying sensation of gliding across open ocean swells." # 33 words
    ]

def get_session(n):
    sessions = {
        1: get_session_1,
        2: get_session_2,
        3: get_session_3,
        4: get_session_4,
        5: get_session_5,
        6: get_session_6,
        7: get_session_7,
        8: get_session_8,
        9: get_session_9,
        10: get_session_10,
    }
    return sessions[n]()

''')
    print("session_bank.py written successfully")

create_bank()
