# Ofgem Smart Export Guarantee (SEG), Agile Plunge Pricing & UK GDPR MPAN Rules

## [OFGEM-SEG-01] Ofgem Smart Export Guarantee (SEG) Eligibility & Metering
Authority: Ofgem

Under the Ofgem Smart Export Guarantee (SEG), licensed electricity suppliers with 150,000 or more domestic customers must offer at least one positive export tariff (> 0 p/kWh) for eligible low-carbon generation up to 5 MW. Generators must hold an export MPAN with half-hourly smart meter capability and MCS (or equivalent) installation certification.

## [UK-AGILE-01] Agile Plunge Pricing & Wind Curtailment Absorption
Authority: GB Wholesale / Half-Hourly Retail Market

During high-wind curtailment windows, GB wholesale and Octopus Agile half-hourly import prices can drop near zero or negative (plunge pricing). Smart EV chargers and home batteries absorb surplus wind energy while respecting local DNO substation headroom and site main fuse thermal current limits.

## [UK-GDPR-01] Smart Meter MPAN UK GDPR Pseudonymisation
Authority: UK GDPR / Smart Energy Code

Half-hourly smart meter MPAN identifiers constitute personal data under UK GDPR. Production telemetry pipelines must pseudonymise raw MPANs using keyed cryptographic hashing (HMAC-SHA256) before analytics persistence or dead-letter quarantine storage.
