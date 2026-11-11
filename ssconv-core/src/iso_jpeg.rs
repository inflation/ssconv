//! ISO 21496-1 JPEG APP2 metadata and CIPA DC-007 MPF packaging.
//!
//! The encoder uses one set of parameters shared by all RGB gain-map channels:
//! SDR base, base colour space, linear gain gamma, and offsets of 1/64.

use crate::SsconvError;

const ISO_NAMESPACE: &[u8] = b"urn:iso:std:iso:ts:21496:-1\0";
const XMP_NAMESPACE: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";

pub(crate) struct Metadata {
    pub minimum: f64,
    pub maximum: f64,
    pub headroom: f64,
}

impl Metadata {
    fn iso_payload(&self) -> Result<Vec<u8>, SsconvError> {
        let mut bytes = ISO_NAMESPACE.to_vec();
        // minimum_version, writer_version = 0. Bit 6: use base colour space.
        // Shared channel parameters; no common denominator or backward flag.
        bytes.extend_from_slice(&[0, 0, 0, 0, 0x40]);
        for value in [
            0.0,
            self.headroom,
            self.minimum,
            self.maximum,
            1.0,
            1.0 / 64.0,
            1.0 / 64.0,
        ] {
            fraction(&mut bytes, value)?;
        }
        Ok(bytes)
    }

    fn xmp(&self) -> String {
        packet(&format!(
            r#"<rdf:Description rdf:about="" xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/" hdrgm:Version="1.0" hdrgm:BaseRenditionIsHDR="False" hdrgm:GainMapMin="{}" hdrgm:GainMapMax="{}" hdrgm:Gamma="1" hdrgm:OffsetSDR="0.015625" hdrgm:OffsetHDR="0.015625" hdrgm:HDRCapacityMin="0" hdrgm:HDRCapacityMax="{}"/>"#,
            self.minimum, self.maximum, self.headroom
        ))
    }
}

// Fixed millionths resolve the minimum nonzero headroom/gain range (1e-6)
// and all log2 gains reachable from finite f32 inputs without saturating.
#[allow(clippy::cast_possible_truncation)]
fn fraction(bytes: &mut Vec<u8>, value: f64) -> Result<(), SsconvError> {
    let numerator = (value * 1_000_000.0).round();
    if !numerator.is_finite() || numerator < f64::from(i32::MIN) || numerator > f64::from(i32::MAX)
    {
        return Err(invalid("ISO Gain Map fraction exceeds supported range"));
    }
    bytes.extend_from_slice(&(numerator as i32).to_be_bytes());
    bytes.extend_from_slice(&1_000_000_u32.to_be_bytes());
    Ok(())
}

pub(crate) fn encode(
    base: &[u8],
    gain: &[u8],
    metadata: &Metadata,
) -> Result<Vec<u8>, SsconvError> {
    if !base.starts_with(&[0xff, 0xd8]) || !gain.starts_with(&[0xff, 0xd8]) {
        return Err(invalid("ISO Gain Map requires JPEG images"));
    }
    let mut secondary = vec![0xff, 0xd8];
    app(&mut secondary, 0xe2, &metadata.iso_payload()?)?;
    xmp(&mut secondary, &metadata.xmp())?;
    secondary.extend_from_slice(&gain[2..]);

    let mut primary_metadata = Vec::new();
    let mut version = ISO_NAMESPACE.to_vec();
    version.extend_from_slice(&[0; 4]);
    app(&mut primary_metadata, 0xe2, &version)?;
    let directory = packet(&format!(
        r#"<rdf:Description rdf:about="" xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/" xmlns:Container="http://ns.google.com/photos/1.0/container/" xmlns:Item="http://ns.google.com/photos/1.0/container/item/" hdrgm:Version="1.0"><Container:Directory><rdf:Seq><rdf:li rdf:parseType="Resource"><Container:Item Item:Semantic="Primary" Item:Mime="image/jpeg"/></rdf:li><rdf:li rdf:parseType="Resource"><Container:Item Item:Semantic="GainMap" Item:Mime="image/jpeg" Item:Length="{}"/></rdf:li></rdf:Seq></Container:Directory></rdf:Description>"#,
        secondary.len()
    ));
    xmp(&mut primary_metadata, &directory)?;
    // The MPF APP2 marker is always inserted immediately after SOI. Its TIFF
    // header therefore starts at file byte 10; all MP image offsets use that base.
    let mpf_size = mpf(0, 0)?.len() + 4;
    let primary_size = base
        .len()
        .checked_add(primary_metadata.len())
        .and_then(|n| n.checked_add(mpf_size))
        .ok_or_else(|| invalid("ISO Gain Map JPEG size overflow"))?;
    let mut result = vec![0xff, 0xd8];
    app(&mut result, 0xe2, &mpf(primary_size, secondary.len())?)?;
    result.extend_from_slice(&primary_metadata);
    result.extend_from_slice(&base[2..]);
    result.extend_from_slice(&secondary);
    Ok(result)
}

fn mpf(primary_size: usize, secondary_size: usize) -> Result<Vec<u8>, SsconvError> {
    let primary_size =
        u32::try_from(primary_size).map_err(|_| invalid("primary JPEG exceeds MPF size limit"))?;
    let secondary_size = u32::try_from(secondary_size)
        .map_err(|_| invalid("gain map JPEG exceeds MPF size limit"))?;
    let mut bytes = b"MPF\0MM\0\x2a\0\0\0\x08".to_vec();
    bytes.extend_from_slice(&3_u16.to_be_bytes());
    // TIFF IFD entries: version (UNDEFINED), image count (LONG), MP entries.
    for (tag, kind, count, value) in [
        (0xb000_u16, 7_u16, 4_u32, u32::from_be_bytes(*b"0100")),
        (0xb001, 4, 1, 2),
        (0xb002, 7, 32, 50),
    ] {
        bytes.extend_from_slice(&tag.to_be_bytes());
        bytes.extend_from_slice(&kind.to_be_bytes());
        bytes.extend_from_slice(&count.to_be_bytes());
        bytes.extend_from_slice(&value.to_be_bytes());
    }
    bytes.extend_from_slice(&0_u32.to_be_bytes()); // no next IFD
    for (attribute, size, offset) in [
        (0x2003_0000_u32, primary_size, 0_u32),
        (0, secondary_size, primary_size.saturating_sub(10)),
    ] {
        bytes.extend_from_slice(&attribute.to_be_bytes());
        bytes.extend_from_slice(&size.to_be_bytes());
        bytes.extend_from_slice(&offset.to_be_bytes());
        bytes.extend_from_slice(&[0; 4]); // no dependent images
    }
    Ok(bytes)
}

fn app(output: &mut Vec<u8>, marker: u8, payload: &[u8]) -> Result<(), SsconvError> {
    let length = payload
        .len()
        .checked_add(2)
        .and_then(|n| u16::try_from(n).ok())
        .ok_or_else(|| invalid("JPEG metadata exceeds APP marker size limit"))?;
    output.extend_from_slice(&[0xff, marker]);
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(payload);
    Ok(())
}

fn packet(description: &str) -> String {
    format!(
        r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">{description}</rdf:RDF></x:xmpmeta>"#
    )
}

fn xmp(output: &mut Vec<u8>, xml: &str) -> Result<(), SsconvError> {
    let mut payload = XMP_NAMESPACE.to_vec();
    payload.extend_from_slice(xml.as_bytes());
    app(output, 0xe1, &payload)
}

fn invalid(message: &str) -> SsconvError {
    SsconvError::InvalidImage {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_metadata_has_expected_fraction_layout() {
        let metadata = Metadata {
            minimum: -1.0,
            maximum: 2.0,
            headroom: 3.0,
        };
        let bytes = metadata.iso_payload().unwrap();
        // ISO JPEG payload: versions, flags, two headroom rationals, followed by
        // min/max/gamma/base-offset/alternate-offset. Each is BE numerator/denominator.
        let mut expected = vec![0, 0, 0, 0, 0x40];
        for numerator in [
            0_i32, 3_000_000, -1_000_000, 2_000_000, 1_000_000, 15_625, 15_625,
        ] {
            expected.extend_from_slice(&numerator.to_be_bytes());
            expected.extend_from_slice(&1_000_000_u32.to_be_bytes());
        }
        assert_eq!(&bytes[ISO_NAMESPACE.len()..], expected);
    }

    #[test]
    fn oversized_metadata_is_rejected() {
        assert!(app(&mut Vec::new(), 0xe2, &vec![0; 65_534]).is_err());
        assert!(fraction(&mut Vec::new(), f64::NAN).is_err());
        assert!(fraction(&mut Vec::new(), f64::MAX).is_err());
    }
}
