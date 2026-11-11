//! Test-only wire reader: locate images through MPF, read ISO APP2 fractions,
//! and leave pixel decoding to the image crate. Does not call the writer.

pub struct Metadata {
    pub gain_map_min: [f64; 3],
    pub gain_map_max: [f64; 3],
    pub base_offset: [f64; 3],
    pub alternate_offset: [f64; 3],
    pub alternate_hdr_headroom: f64,
}

pub struct IsoJpeg<'a> {
    pub base: &'a [u8],
    pub gain: &'a [u8],
    pub metadata: Metadata,
}

impl<'a> IsoJpeg<'a> {
    pub fn parse(bytes: &'a [u8]) -> Self {
        let primary_markers = markers(bytes);
        let (_, mpf, start) = primary_markers
            .iter()
            .find(|(kind, data, _)| *kind == 0xe2 && data.starts_with(b"MPF\0"))
            .unwrap();
        let tiff = &mpf[4..];
        assert_eq!(&tiff[..4], b"MM\0\x2a");
        let ifd = usize::try_from(u32_at(tiff, 4)).unwrap();
        let mut entries = None;
        for i in 0..usize::from(u16_at(tiff, ifd)) {
            let entry = ifd + 2 + i * 12;
            match u16_at(tiff, entry) {
                0xb001 => assert_eq!(u32_at(tiff, entry + 8), 2),
                0xb002 => {
                    assert_eq!(u32_at(tiff, entry + 4), 32);
                    entries = Some(usize::try_from(u32_at(tiff, entry + 8)).unwrap());
                }
                _ => {}
            }
        }
        let entries = entries.unwrap();
        let primary_length = usize::try_from(u32_at(tiff, entries + 4)).unwrap();
        assert_eq!(u32_at(tiff, entries + 8), 0);
        let secondary_length = usize::try_from(u32_at(tiff, entries + 20)).unwrap();
        let secondary_offset = start + 4 + usize::try_from(u32_at(tiff, entries + 24)).unwrap();
        assert_eq!(secondary_offset, primary_length);
        assert_eq!(secondary_offset + secondary_length, bytes.len());
        let base = &bytes[..primary_length];
        let gain = &bytes[secondary_offset..];
        assert_eq!(&base[base.len() - 2..], [0xff, 0xd9]);
        assert_eq!(&gain[gain.len() - 2..], [0xff, 0xd9]);
        let namespace = b"urn:iso:std:iso:ts:21496:-1\0";
        let (_, version, _) = primary_markers
            .iter()
            .find(|(kind, data, _)| *kind == 0xe2 && data.starts_with(namespace))
            .unwrap();
        assert_eq!(&version[namespace.len()..], [0; 4]);
        let gain_markers = markers(gain);
        let (_, iso, _) = gain_markers
            .iter()
            .find(|(kind, data, _)| *kind == 0xe2 && data.starts_with(namespace))
            .unwrap();
        let iso = &iso[namespace.len()..];
        assert_eq!(&iso[..5], [0, 0, 0, 0, 0x40]);
        assert_eq!(iso.len(), 61);
        let fields: Vec<f64> = (0..7)
            .map(|i| {
                let position = 5 + i * 8;
                let denominator = u32_at(iso, position + 4);
                assert_ne!(denominator, 0);
                let numerator = if [2, 3, 5, 6].contains(&i) {
                    f64::from(i32::from_be_bytes(
                        iso[position..position + 4].try_into().unwrap(),
                    ))
                } else {
                    f64::from(u32_at(iso, position))
                };
                numerator / f64::from(denominator)
            })
            .collect();
        assert_eq!(fields[0], 0.0);
        assert!(fields[1] > 0.0);
        assert!(fields[2] <= fields[3]);
        assert_eq!(fields[4], 1.0);
        Self {
            base,
            gain,
            metadata: Metadata {
                gain_map_min: [fields[2]; 3],
                gain_map_max: [fields[3]; 3],
                base_offset: [fields[5]; 3],
                alternate_offset: [fields[6]; 3],
                alternate_hdr_headroom: fields[1],
            },
        }
    }
}

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_be_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn markers(bytes: &[u8]) -> Vec<(u8, &[u8], usize)> {
    assert_eq!(&bytes[..2], [0xff, 0xd8]);
    let mut result = Vec::new();
    let mut position = 2;
    while bytes[position..position + 2] != [0xff, 0xda] {
        assert_eq!(bytes[position], 0xff);
        let length = usize::from(u16_at(bytes, position + 2));
        result.push((
            bytes[position + 1],
            &bytes[position + 4..position + 2 + length],
            position + 4,
        ));
        position += length + 2;
    }
    result
}
