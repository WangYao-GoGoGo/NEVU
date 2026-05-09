tr_ag = {
        "Cognitive.IdentifyCategorize.Unspecified": [
        "IdentifiedObject",  # The object being identified
        "Identifier",  # The subject performing the identification
        "IdentifiedRole",  # The result or category of the identification
        "Place",  # The location where the event occurs
        "Time",  # The time when the event occurs
        "Purpose",  # The purpose of the identification
        "Outcome"  # The result or consequence of the identification
    ],
    "Cognitive.Inspection.SensoryObserve": [
        "Observer", # The observer
        "ObservedEntity", # The object being observed
        "Instrument", # The tools or means used for observation
        "Place", # The location where the observation occurs
        "Time", # The time when the observation occurs
        "Purpose", # The purpose of the observation
        "Outcome", # The result or impact of the observation
        "EmotionalState", # The emotional state of the observer
        "Impact" # The impact on society or the environment
    ],
    "Conflict.Attack.Unspecified": [
        "Place",           # The location where the event occurs
        "Attacker",        # The entity initiating the attack (individual or organization)
        "Target",          # The object being attacked (individual, organization, or object)
        "Instrument",      # The tools or means used for the attack
        "Time",            # The time when the attack occurs
        "Motive",          # The motive or reason for the attack
        "Outcome",         # The result of the attack (e.g., casualties, damage)
        "Casualties",      # The number of casualties resulting from the event
        "Impact"           # The impact of the attack on society or the environment
    ],
    "Life.Injure.Unspecified": [
        "Victim",          # The injured individual directly affected by the event
        "Injurer",         # The entity causing the injury (individual or entity)
        "Instrument",      # The tools or means used to cause the injury
        "BodyPart",        # The body part that was injured
        "Time",            # The time when the injury occurred
        "Place",           # The location where the injury occurred
        "Severity",        # The severity of the injury (e.g., minor, severe)
        "Outcome",         # The result of the event (e.g., recovery, disability, death)
        "Motive",          # The motive or reason for the act leading to the injury
        "Impact"           # The impact on society, family, or the environment
    ],
    "Conflict.Attack.DetonateExplode": [
        "ExplosiveDevice",  # The explosive device (e.g., bomb, landmine)
        "Target",           # The target of the attack (e.g., building, crowd)
        "Attacker",         # The entity initiating the attack (individual or organization)
        "Place",            # The location where the explosion occurred
        "Time",             # The time when the explosion occurred
        "Motive",           # The motive or reason for the attack (e.g., terrorism, political goals)
        "Casualties",       # The number of casualties (including deaths and injuries)
        "Outcome",          # The result of the explosion (e.g., level of destruction, success of the target)
        "Impact"            # The broader impact on society, economy, or environment
    ],
    "Justice.ChargeIndict.Unspecified": [
        "Defendant",        # The individual or entity being accused (suspect or defendant)
        "JudgeCourt",       # The judge or court responsible for the case
        "Prosecutor",       # The party bringing the accusation (prosecutor or plaintiff)
        "Place",            # The location where the charge or trial takes place
        "Time",             # The time when the charge or indictment occurs
        "Charge",           # The specific crime or accusation (e.g., theft, fraud)
        "Motive",           # The motive behind the charge (e.g., seeking justice, retaliation)
        "Outcome",          # The result of the charge (e.g., trial, dismissal)
        "Evidence",         # The critical evidence supporting the charge
        "Impact"            # The broader impact on the defendant, prosecutor, or society
    ],
    "Justice.ArrestJailDetain.Unspecified": [
        "Detainee",        # The individual being detained or arrested
        "Jailer",          # The authority or individual responsible for the detention or arrest
        "Place",           # The location where the detention or arrest takes place
        "Time",            # The time of the detention or arrest
        "Reason",          # The reason or justification for the detention or arrest
        "Motive",          # The underlying motive behind the detention (e.g., law enforcement, political reasons)
        "Outcome",         # The immediate result of the detention (e.g., released, held for trial)
        "Evidence",        # The evidence or basis for the detention or arrest
        "Impact"           # The broader impact on the individual, their family, or society
    ],
    "Justice.Convict.Unspecified": [
        "Defendant",  # The individual being convicted
        "JudgeCourt",  # The judge or court delivering the conviction
        "Time",  # The time when the conviction is issued
        "Place",  # The location of the trial or conviction
        "Charge",  # The specific crime or offense for which the conviction is made
        "Sentence",  # The punishment or penalty assigned (e.g., imprisonment, fine)
        "Evidence",  # The evidence that led to the conviction
        "Outcome",  # The broader result of the conviction (e.g., appeal, sentence served)
        "Impact"  # The social or personal impact of the conviction on the defendant or community
    ],
    "Justice.InvestigateCrime.Unspecified": [
        "Investigator",
        "Defendant",
        "Place",
        "Observer",
        "ObservedEntity"
    ],
    "Contact.Contact.Unspecified": [
        "Participant",  # Individuals or entities involved in the contact
        "Place",  # The location where the contact takes place
        "Time",  # The time when the contact occurs
        "Topic",  # The subject or focus of the contact
        "Purpose",  # The intention or reason behind the contact
        "Outcome",  # The result or conclusion of the contact
        "ModeOfContact",  # The medium or method used for the contact (e.g., in-person, phone, email)
        "Impact"  # The broader social, personal, or systemic effects of the contact
    ],
    "Life.Die.Unspecified": [
        "Victim",  # The individual who died
        "Place",  # The location where the death occurred
        "Time",  # The time of death
        "Killer",  # The individual or entity responsible for the death, if applicable
        "CauseOfDeath",  # The reason or method leading to the death (e.g., illness, accident, homicide)
        "Motive",  # The motive behind the death, if applicable (e.g., self-defense, malice)
        "Impact",  # The broader effects of the death on family, society, or the environment
        "Witness",  # Any individual who observed or discovered the death
        "Outcome",  # The immediate or long-term result of the death (e.g., investigation, community response)
        "Severity",  # The scale or seriousness of the event (e.g., isolated death, mass casualty)
    ],
    "ArtifactExistence.DamageDestroyDisableDismantle.Damage": [
        "Artifact",  # The object or structure that was damaged
        "Damager",  # The individual, entity, or force responsible for the damage
        "Place",  # The location where the damage occurred
        "Time",  # The time when the damage occurred
        "Instrument",  # The tool or method used to cause the damage, if applicable
        "Cause",  # The reason or cause behind the damage (e.g., accident, vandalism, natural disaster)
        "Severity",  # The extent or seriousness of the damage
        "Impact",  # The broader effects of the damage on society, the environment, or the economy
        "Outcome",  # The immediate or long-term result of the damage (e.g., repair, abandonment)
    ],
    "Control.ImpedeInterfereWith.Unspecified": [
        "Impeder",  # The individual or entity causing the interference
        "Place",  # The location where the interference occurs
        "Time",  # The time when the interference happens
        "Target",  # The individual, entity, or process being impeded
        "Method",  # The method or means used to cause the interference
        "Reason",  # The reason or justification for the interference
        "Outcome",  # The result or consequence of the interference
        "Impact"  # The broader effects of the interference on society, individuals, or systems
    ],
    "GenericCrime.GenericCrime.GenericCrime": [
        "Place",  # The location where the crime occurred
        "Perpetrator",  # The individual or entity responsible for the crime
        "Victim",  # The individual or entity harmed by the crime
        "Time",  # The time when the crime occurred
        "CrimeType",  # The specific type of crime (e.g., theft, assault, fraud)
        "Motive",  # The reason or intent behind committing the crime
        "Outcome",  # The result or consequences of the crime (e.g., injuries, losses)
        "Evidence",  # The evidence supporting the identification of the crime or perpetrator
        "Impact"  # The broader effects of the crime on society, community, or individuals
    ],
    "Movement.Transportation.Unspecified": [
        "Vehicle",  # The mode of transportation used (e.g., car, train, airplane)
        "PassengerArtifact",  # The individual(s) or items being transported
        "Transporter",  # The individual or entity responsible for conducting the transportation
        "Origin",  # The starting point of the transportation
        "Destination",  # The endpoint or goal of the transportation
        "Time",  # The time when the transportation occurs
        "Purpose",  # The reason for the transportation (e.g., travel, delivery, evacuation)
        "Outcome",  # The result of the transportation (e.g., successful arrival, delays, accidents)
        "Impact"  # The broader effects of the transportation on society, the environment, or individuals

    ],
    "Contact.Contact.Broadcast": [
        "Place",  # The location from which the broadcast originates
        "Communicator",  # The individual or entity delivering the broadcast
        "Topic",  # The subject or focus of the broadcast
        "Instrument",  # The medium or tool used for the broadcast (e.g., television, radio, internet)
        "Recipient",  # The intended audience or group receiving the broadcast
        "Time",  # The time when the broadcast occurs
        "Purpose",  # The reason or intent behind the broadcast (e.g., education, advertising, emergency alert)
        "Outcome",  # The result or effect of the broadcast (e.g., awareness, behavioral change, confusion)
        "Impact"  # The broader effects of the broadcast on society, individuals, or systems
    ],
    "ArtifactExistence.DamageDestroyDisableDismantle.Destroy": [
        "Artifact",  # The object or structure that was destroyed
        "Instrument",  # The tool, method, or means used for destruction
        "Place",  # The location where the destruction occurred
        "Destroyer",  # The individual, entity, or force responsible for the destruction
        "Time",  # The time when the destruction occurred
        "Cause",  # The reason or trigger for the destruction (e.g., accident, deliberate act, natural disaster)
        "Outcome",  # The immediate result of the destruction (e.g., loss of function, structural collapse)
        "Severity",  # The extent or seriousness of the destruction
        "Impact"  # The broader effects of the destruction on society, environment, or individuals

    ],
    "Medical.Intervention.Unspecified": [
        "Patient",  # The individual receiving the medical intervention
        "Treater",  # The healthcare professional or entity providing the treatment
        "Place",  # The location where the intervention occurs (e.g., hospital, clinic)
        "Time",  # The time when the intervention occurs
        "InterventionType",  # The type of medical intervention (e.g., surgery, medication, therapy)
        "Purpose",  # The reason or goal for the intervention (e.g., cure, relief, prevention)
        "Outcome",  # The result of the intervention (e.g., recovery, complications, death)
        "Severity",  # The seriousness of the medical condition or intervention
        "Impact"  # The broader effects of the intervention on the patient, family, or society

    ],
    "Conflict.Demonstrate.DemonstrateWithViolence": [
        "Demonstrator",  # The individuals or groups participating in the violent demonstration
        "Place",  # The location where the demonstration occurs
        "Time",  # The time when the demonstration takes place
        "Cause",  # The reason or trigger for the demonstration
        "Method",  # The means or tactics used during the demonstration (e.g., rioting, vandalism)
        "Target",  # The individuals, groups, or entities targeted by the violence
        "Outcome",  # The result of the demonstration (e.g., injuries, property damage, policy changes)
        "Casualties",  # The number of people injured or killed during the demonstration
        "Impact"  # The broader societal, economic, or political effects of the violent demonstration

    ],
    "Conflict.Demonstrate.Unspecified": [
        "Demonstrator",  # The individuals or groups participating in the demonstration
        "Topic",  # The subject or issue that the demonstration is about
        "Target",  # The individuals, groups, or entities that the demonstration is directed at
        "Place",  # The location where the demonstration occurs
        "Time",  # The time when the demonstration takes place
        "Cause",  # The reason or trigger for the demonstration
        "Method",  # The means or tactics used during the demonstration (e.g., marches, sit-ins)
        "Outcome",  # The result of the demonstration (e.g., policy changes, arrests)
        "Impact"  # The broader societal, economic, or political effects of the demonstration
    ],
    "Contact.ThreatenCoerce.Unspecified": [
        "Communicator",  # The individual or entity making the threat or coercion
        "Recipient",  # The individual or entity receiving the threat or coercion
        "Place",  # The location where the threat or coercion occurs
        "Time",  # The time when the threat or coercion occurs
        "Method",  # The means or manner of delivering the threat or coercion (e.g., verbal, written, physical)
        "Reason",  # The intent or purpose behind the threat or coercion (e.g., to gain compliance, revenge)
        "Outcome",  # The result of the threat or coercion (e.g., compliance, resistance, legal action)
        "Impact"  # The broader social, personal, or systemic effects of the threat or coercion

    ],
    "Contact.RequestCommand.Broadcast": [
        "Communicator",  # The individual or entity issuing the request or command
        "Recipient",  # The intended audience or group receiving the request or command
        "Place",  # The location from which the broadcast is made
        "Time",  # The time when the broadcast occurs
        "Topic",  # The subject or content of the request or command
        "Method",  # The medium or channel used for the broadcast (e.g., radio, television, social media)
        "Purpose",  # The reason or intent behind the broadcast (e.g., to inform, to mobilize action)
        "Outcome",  # The result of the broadcast (e.g., compliance, resistance, misunderstanding)
        "Impact"  # The broader social, economic, or political effects of the broadcast
    ],
    "Contact.Contact.Meet": [
        "Participant",  # The individuals or entities involved in the meeting
        "Topic",  # The subject or agenda of the meeting
        "Place",  # The location where the meeting takes place
        "Time",  # The time when the meeting occurs
        "Purpose",  # The reason or intent behind the meeting (e.g., negotiation, collaboration)
        "Outcome",  # The result or conclusion of the meeting (e.g., agreement, conflict, no resolution)
        "Impact"  # The broader effects of the meeting on individuals, relationships, or society
    ],
    "Movement.Transportation.Evacuation": [
        "Transporter",  # The individual or entity responsible for conducting the evacuation
        "PassengerArtifact",  # The individuals, groups, or items being evacuated
        "Origin",  # The starting point of the evacuation
        "Destination",  # The endpoint or safe location for the evacuation
        "Time",  # The time when the evacuation occurs
        "Cause",  # The reason or trigger for the evacuation (e.g., natural disaster, conflict)
        "Method",  # The mode of transportation used (e.g., buses, helicopters, boats)
        "Outcome",  # The result of the evacuation (e.g., safe relocation, delays, casualties)
        "Impact"  # The broader effects of the evacuation on society, the environment, or the evacuees
    ],
    "Justice.Acquit.Unspecified": [
        "Defendant",  # The individual or entity being acquitted
        "JudgeCourt",  # The judge or court delivering the acquittal
        "Time",  # The time when the acquittal occurs
        "Place",  # The location where the acquittal is announced
        "Charge",  # The specific crime or accusation the defendant was acquitted of
        "Reason",  # The reason for the acquittal (e.g., lack of evidence, procedural error)
        "Outcome",  # The result of the acquittal (e.g., release from custody, reputation restored)
        "Impact"  # The broader effects of the acquittal on society, the legal system, or the individual
    ],
    "ArtifactExistence.ManufactureAssemble.Unspecified": [
        "Components",  # The individual parts or materials used in the manufacturing or assembly process
        "Artifact",  # The finished product or artifact created through manufacturing or assembly
        "Place",  # The location where the manufacturing or assembly takes place
        "ManufacturerAssembler",  # The individual, group, or entity responsible for the process
        "Time",  # The time when the manufacturing or assembly occurs
        "Method",  # The specific process or technique used for manufacturing or assembly
        "Purpose",  # The reason or goal for creating the artifact (e.g., commercial, functional, artistic)
        "Outcome",  # The result of the manufacturing or assembly process (e.g., completed product, defective product)
        "Impact"  # The broader societal, economic, or environmental effects of the manufacturing or assembly process
    ],
    "ArtifactExistence.DamageDestroyDisableDismantle.Dismantle": [
        "Components",  # The individual parts or materials resulting from the dismantling process
        "Instrument",  # The tools or methods used to dismantle the artifact
        "Place",  # The location where the dismantling occurs
        "Artifact",  # The object or structure being dismantled
        "Dismantler",  # The individual, group, or entity responsible for the dismantling process
        "Time",  # The time when the dismantling occurs
        "Purpose",  # The reason or goal for dismantling the artifact (e.g., recycling, decommissioning)
        "Outcome",  # The result of the dismantling process (e.g., materials salvaged, artifact destroyed)
        "Impact"  # The broader effects of the dismantling on society, the environment, or the economy

    ],
    "Justice.Sentence.Unspecified": [
        "Defendant",  # The individual or entity being sentenced
        "JudgeCourt",  # The judge or court delivering the sentence
        "Place",  # The location where the sentencing occurs
        "Time",  # The time when the sentence is delivered
        "Charge",  # The specific crime or offense for which the sentence is being given
        "Sentence",  # The punishment or penalty assigned (e.g., imprisonment, fine, community service)
        "Reason",  # The justification or reasoning behind the sentence (e.g., deterrence, rehabilitation)
        "Outcome",  # The result of the sentencing (e.g., defendant complies, appeals)
        "Impact"  # The broader societal, personal, or systemic effects of the sentencing
    ],
    "Justice.TrialHearing.Unspecified": [
        "Defendant",  # The individual or entity standing trial
        "Place",  # The location where the trial or hearing takes place
        "Prosecutor",  # The individual or entity representing the prosecution
        "JudgeCourt",  # The judge or court overseeing the trial or hearing
        "Time",  # The time when the trial or hearing occurs
        "Charge",  # The specific crime or offense being addressed in the trial
        "Outcome",  # The result or conclusion of the trial or hearing (e.g., conviction, acquittal, adjournment)
        "Reason",  # The purpose or intent behind the trial (e.g., to determine guilt, resolve a dispute)
        "EvidencePresented",  # The evidence introduced during the trial or hearing
        "Impact"  # The broader societal, legal, or personal effects of the trial or hearing
    ],
    "Transaction.ExchangeBuySell.Unspecified": [
        "Recipient",  # The individual or entity receiving the acquired entity (e.g., buyer)
        "AcquiredEntity",  # The object, service, or asset being exchanged or acquired
        "Giver",  # The individual or entity transferring the acquired entity (e.g., seller)
        "PaymentBarter",  # The payment, barter, or consideration provided in exchange
        "Place",  # The location where the transaction occurs
        "Time",  # The time when the transaction occurs
        "Method",  # The method of transaction (e.g., cash, credit, online payment)
        "Purpose",  # The reason or intent behind the transaction (e.g., personal use, investment)
        "Outcome",  # The result of the transaction (e.g., completed, canceled, disputed)
        "Impact"  # The broader effects of the transaction on society, the market, or individuals
    ],
    "Movement.Transportation.PreventPassage": [
        "Destination",  # The intended endpoint of the blocked or prevented passage
        "Preventer",  # The individual, group, or entity responsible for preventing the passage
        "Origin",  # The starting point of the transportation being obstructed
        "Vehicle",  # The mode of transportation being obstructed (e.g., car, train, ship)
        "PassengerArtifact", # The individuals, groups, or items being transported and prevented from reaching the destination
        "Transporter",  # The individual or entity responsible for conducting the transportation
        "Place",  # The location where the obstruction occurs
        "Time",  # The time when the passage is prevented
        "Method",  # The means or tactics used to prevent the passage (e.g., barriers, protests, blockades)
        "Reason",  # The reason or motive for preventing the passage (e.g., safety, protest, control)
        "Outcome",  # The result of the obstruction (e.g., successful prevention, rerouting, conflict)
        "Impact"  # The broader effects of the obstruction on society, the environment, or the individuals involved
    ],
    "Contact.Contact.Correspondence": [
        "Participant",  # The individuals or entities involved in the correspondence
        "Place",  # The location(s) where the correspondence is sent or received
        "Topic",  # The subject or focus of the correspondence
        "Time",  # The time when the correspondence occurs
        "Method",  # The medium or channel used for the correspondence (e.g., email, letter, text)
        "Purpose",  # The intent or reason for the correspondence (e.g., to inform, negotiate, collaborate)
        "Outcome", # The result or conclusion of the correspondence (e.g., agreement, misunderstanding, continued discussion)
        "Impact"  # The broader effects of the correspondence on relationships, organizations, or society
    ],
    "Contact.ThreatenCoerce.Broadcast": [
        "Communicator",  # The individual or entity issuing the threat or coercion through the broadcast
        "Recipient",  # The intended audience or group targeted by the broadcast
        "Place",  # The location from which the broadcast is made
        "Time",  # The time when the broadcast occurs
        "Method",  # The medium or channel used for the broadcast (e.g., television, radio, social media)
        "Reason",  # The intent or purpose behind the threat or coercion (e.g., to intimidate, demand compliance)
        "Outcome",  # The result of the broadcast (e.g., compliance, resistance, legal action)
        "Impact"  # The broader effects of the broadcast on society, individuals, or systems
    ],
    "Contact.RequestCommand.Unspecified": [
        "Communicator",  # The individual or entity making the request or issuing the command
        "Recipient",  # The intended audience or group receiving the request or command
        "Place",  # The location where the request or command is issued or received
        "Time",  # The time when the request or command is made
        "Topic",  # The subject or content of the request or command
        "Method",  # The medium or channel used for communication (e.g., verbal, written, broadcast)
        "Purpose",  # The reason or intent behind the request or command (e.g., to obtain information, enforce action)
        "Outcome",  # The result or response to the request or command (e.g., compliance, resistance, misunderstanding)
        "Impact"  # The broader effects of the request or command on individuals, relationships, or society
    ],
    "Conflict.Defeat.Unspecified": [
        "Victor",  # The individual, group, or entity achieving the victory
        "Defeated",  # The individual, group, or entity that was defeated
        "Place",  # The location where the conflict or defeat occurred
        "Time",  # The time when the defeat occurred
        "Cause",  # The reason or factors leading to the defeat (e.g., strategic errors, superior force)
        "Method", # The tactics or means used by the victor to achieve the defeat (e.g., military strategy, negotiations)
        "Outcome",  # The immediate result of the defeat (e.g., surrender, loss of resources)
        "Impact"  # The broader societal, economic, or political effects of the defeat
    ],
    "Life.Infect.Unspecified": [
        "Victim",  # The individual or group infected by the pathogen
        "InfectiousAgent",  # The pathogen or cause of the infection (e.g., virus, bacteria, fungus)
        "Source",  # The origin of the infection (e.g., another infected individual, environment)
        "Place",  # The location where the infection occurred or was identified
        "Time",  # The time when the infection occurred or was detected
        "TransmissionMethod",  # The means by which the infection was transmitted (e.g., airborne, contact, vector)
        "Symptoms",  # The symptoms exhibited by the victim due to the infection
        "Severity",  # The seriousness of the infection (e.g., mild, severe, life-threatening)
        "Outcome",  # The result of the infection (e.g., recovery, death, long-term complications)
        "Impact"  # The broader societal, economic, or environmental effects of the infection
    ],
    "Cognitive.Research.Unspecified": [
        "Researcher",  # The individual or group conducting the research
        "Subject",  # The object, phenomenon, or topic being studied
        "Place",  # The location where the research is conducted
        "Time",  # The time when the research takes place
        "Method",  # The approach or methodology used in the research (e.g., experiments, surveys, observation)
        "Purpose",  # The intent or goal of the research (e.g., to gain knowledge, solve a problem)
        "Outcome",  # The result or findings of the research
        "EthicsConsidered",  # Ethical considerations involved in the research (e.g., informed consent, privacy)
        "Impact"  # The broader effects of the research on society, science, or the environment
    ],
    "Disaster.Crash.Unspecified": [
        "Vehicle",  # The vehicle involved in the crash (e.g., car, airplane, train)
        "Place",  # The location where the crash occurred
        "CrashObject", # The object or entity with which the vehicle collided (e.g., another vehicle, structure, terrain)
        "Time",  # The time when the crash occurred
        "Cause",  # The reason or factors leading to the crash (e.g., mechanical failure, human error)
        "Severity",  # The seriousness of the crash (e.g., minor damage, major destruction, casualties)
        "Casualties",  # The number of people injured or killed in the crash
        "Outcome",  # The immediate result of the crash (e.g., vehicle destroyed, traffic disruption)
        "Impact"  # The broader societal, economic, or environmental effects of the crash
    ],
    "ArtifactExistence.DamageDestroyDisableDismantle.Unspecified": [
        "Artifact",  # The object or structure that was damaged, destroyed, disabled, or dismantled
        "Instrument",  # The tool, method, or means used in the action
        "DamagerDestroyer",  # The individual, group, or force responsible for the action
        "Place",  # The location where the action occurred
        "Time",  # The time when the action took place
        "Cause",  # The reason or trigger for the action (e.g., accident, deliberate act)
        "Outcome",  # The result of the action (e.g., partial damage, complete destruction, materials salvaged)
        "Severity",  # The extent or seriousness of the action (e.g., minor damage, major destruction)
        "Impact"  # The broader effects of the action on society, the environment, or individuals
    ],
    "Movement.Transportation.IllegalTransportation": [
        "PassengerArtifact",  # The individuals, goods, or contraband being illegally transported
        "Destination",  # The intended endpoint or location of the illegal transportation
        "Transporter",  # The individual or entity responsible for conducting the illegal transportation
        "Vehicle",  # The mode of transportation used for the illegal activity (e.g., car, boat, plane)
        "Place",  # The location where the illegal transportation is initiated, intercepted, or occurs
        "Time",  # The time when the illegal transportation occurs
        "Method",  # The means or tactics used to evade detection (e.g., false documents, hidden compartments)
        "Purpose",  # The intent behind the illegal transportation (e.g., smuggling, human trafficking, drug trade)
        "Outcome",  # The result of the illegal transportation (e.g., successful delivery, interception by authorities)
        "Impact"  # The broader societal, economic, or environmental effects of the illegal transportation
    ],
    "Contact.ThreatenCoerce.Correspondence": [
        "Recipient",  # The individual or entity receiving the threat or coercion through correspondence
        "Communicator",  # The individual or entity issuing the threat or coercion
        "Place",  # The location where the correspondence is sent or received
        "Time",  # The time when the correspondence occurs
        "Method",  # The medium or channel used for the correspondence (e.g., email, letter, text)
        "Reason",  # The intent or purpose behind the threat or coercion (e.g., extortion, intimidation)
        "Content",  # The specific content of the threat or coercion
        "Outcome",  # The result or reaction to the correspondence (e.g., compliance, reporting to authorities)
        "Impact"  # The broader societal, personal, or systemic effects of the correspondence
    ],
    "Personnel.EndPosition.Unspecified": [
        "Employee",  # The individual leaving the position or employment
        "PlaceOfEmployment",  # The organization or company where the position is being ended
        "Position",  # The specific role or title being vacated
        "Time",  # The time when the position or employment ends
        "Reason",  # The reason for ending the position (e.g., resignation, termination, retirement)
        "Outcome",  # The result of the end of employment (e.g., replacement hired, restructuring)
        "Impact"  # The broader effects on the individual, organization, or society
    ],
    "ArtifactExistence.DamageDestroyDisableDismantle.DisableDefuse": [
        "Artifact",  # The object or device that is disabled or defused
        "Instrument",  # The tool, method, or technique used for disabling or defusing
        "Disabler",  # The individual, group, or entity responsible for the action
        "Place",  # The location where the disabling or defusing occurs
        "Time",  # The time when the action occurs
        "Cause",  # The reason or trigger for disabling or defusing (e.g., safety concerns, deactivation of a threat)
        "Outcome",  # The result of the action (e.g., artifact rendered safe, partially disabled)
        "Impact"  # The broader societal, economic, or environmental effects of the disabling or defusing action
    ],
    "Personnel.StartPosition.Unspecified": [
        "Employee",  # The individual starting the new position or employment
        "PlaceOfEmployment",  # The organization or company where the position is being started
        "Position",  # The specific role or title being assumed
        "Time",  # The time when the position or employment begins
        "Reason",  # The reason or context for starting the position (e.g., new hiring, promotion, transfer)
        "Method", # The process or means through which the position was secured (e.g., job application, internal transfer)
        "Outcome", # The initial result or status after starting the position (e.g., probation period, assigned responsibilities)
        "Impact"  # The broader effects on the individual, team, or organization
    ],
    "Cognitive.TeachingTrainingLearning.Unspecified": [
        "Learner",  # The individual or group receiving the teaching or training
        "TeacherTrainer",  # The individual or entity providing the teaching or training
        "Topic",  # The subject or skill being taught or learned
        "Place",  # The location where the teaching or training takes place
        "Time",  # The time when the teaching or training occurs
        "Method", # The approach or medium used for teaching or training (e.g., lecture, hands-on practice, online course)
        "Purpose",  # The intent or goal of the teaching or training (e.g., skill acquisition, knowledge transfer)
        "Outcome",  # The result of the teaching or training (e.g., skill gained, certification earned)
        "Impact"  # The broader effects of the teaching or training on individuals, communities, or society
    ],
    "Justice.ReleaseParole.Unspecified": [
        "Defendant", # The individual being released on parole.
        "JudgeCourt", # The judge or court responsible for granting the parole.
        "ParoleCondition", # The conditions or restrictions imposed on the parolee.
        "Time", # The time when the parole decision or release takes place.
        "Place", # The location where the parole process or release occurs.
        "Reason", # The justification or reason for granting parole.
        "ReleaseSupervisor", # The person or organization supervising the parolee after release.
        "Victim" # The individual or group impacted by the defendant's original offense, if applicable.
    ],
    "Transaction.Donation.Unspecified": [
        "Recipient", # The individual or organization receiving the donation.
        "Giver", # The individual or organization providing the donation.
        "ArtifactMoney", # The item(s) or money being donated.
        "Purpose", # The intended purpose or use of the donation.
        "Time", # The time when the donation occurs.
        "Place", # The location where the donation takes place.
        "TransactionMethod", # The method used for the donation (e.g., cash, bank transfer, physical delivery).
        "Acknowledgment", # The acknowledgment or confirmation of the donation (e.g., receipt, public recognition).
        "Beneficiary" # The specific group or individuals benefiting from the donation.
    ],
    "Disaster.DiseaseOutbreak.Unspecified": [
        "Place", # The location where the disease outbreak occurs.
        "Disease", # The specific disease responsible for the outbreak.
        "Victim", # The individuals affected by the outbreak.
        "Time", # The time when the outbreak begins or is identified.
        "Cause", # The cause or origin of the outbreak (e.g., pathogen, contamination).
        "CarrierVector", # The carrier or vector spreading the disease (e.g., mosquitoes, air, water).
        "InfectedPopulation", # The number of individuals infected by the disease.
        "MortalityRate", # The percentage or number of deaths caused by the outbreak.
        "PreventiveMeasures", # Measures taken to prevent or control the outbreak (e.g., vaccination, quarantine).
        "ResponsibleOrganization", # The organization managing the outbreak response (e.g., WHO, local government).
        "EconomicImpact", # The economic consequences of the outbreak (e.g., healthcare costs, economic losses).
        "Duration" # The duration of the outbreak from start to resolution.
    ],
    "Contact.RequestCommand.Meet": [
        "Recipient", # The individual or group receiving the request to meet.
        "Communicator", # The individual or group making the request to meet.
        "Time", # The proposed or actual time of the meeting.
        "Place", # The location of the proposed or actual meeting.
        "Purpose", # The purpose or reason for the meeting request.
        "Topic", # The topic to be discussed during the meeting.
        "Method", # The method used to communicate the request (e.g., email, phone).
        "Response" # The response to the meeting request (e.g., accept, decline, reschedule).
    ],
    "Contact.RequestCommand.Correspondence": [
        "Recipient", # The individual or group receiving the correspondence request.
        "Communicator", # The individual or group making the correspondence request.
        "Topic", # The subject or content of the correspondence.
        "Time", # The time when the correspondence is requested or occurs.
        "Method", # The method of correspondence (e.g., email, letter, phone).
        "Purpose", # The purpose or reason for the correspondence.
        "Response", # The response to the correspondence request (e.g., acknowledgment, reply).
        "Place", # The location related to the correspondence (if applicable).
        "Urgency" # The level of urgency for the correspondence (e.g., high, low).
    ]
}

if __name__ == '__main__':
    tr_ag_value = tr_ag
    print(tr_ag_value)