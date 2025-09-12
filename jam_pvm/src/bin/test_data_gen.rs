// src/bin/test_data_gen.rs

// This is a separate, runnable program to generate valid SCALE-encoded hex strings
// for testing your JAM PVM service endpoints with curl.

// NOTE: This program requires the `rand` crate. Add it to your `Cargo.toml`
// under the `[dependencies]` section like this:
// rand = "0.8"

// To run this file, use the command:
// cargo run --bin test_data_gen

// Import necessary crates and types from your project and external libraries.
use ed25519_dalek::{ Signer, SigningKey };
use hex;
use jam_pvm::types::{ AuthCredentials, ServiceCommand }; // Your project's types
use jam_types::{
    AccumulateItem,
    AuthOutput,
    Authorization,
    Authorizer,
    BoundedVec,
    CoreIndex,
    Hash,
    MaxWorkItems,
    Memo,
    PayloadHash,
    RefineContext,
    ServiceId,
    Slot,
    TransferRecord,
    VecSet,
    WorkItem,
    WorkOutput,
    WorkPackage,
    WorkPackageHash,
    WorkPayload,
};
// The Encode trait must be in scope to use the .encode() method.
// We use the one from parity-scale-codec as it's the underlying implementation.
use jam_codec::Encode as CodecEncode; // for jam_pvm::types::* like AuthCredentials, ServiceCommand
use jam_types::Encode as TypesEncode; // for jam_types::* like WorkPackage, AccumulateItem, TransferRecord, Vec<...>
use rand::rngs::OsRng;
use sha2::{ Digest, Sha256 };

fn main() {
    println!("--- Generating Test Data for JAM PVM Service ---\n");

    // --- 1. Data for /authorizer/is_authorized ---
    println!("## Endpoint: /authorizer/is_authorized (Success Case)");

    // Generate a new cryptographic keypair.
    let mut csprng = OsRng;
    // The API requires generating a SigningKey directly.
    let signing_key: SigningKey = SigningKey::generate(&mut csprng);
    let public_key_bytes: [u8; 32] = signing_key.verifying_key().to_bytes();

    // Define the payload that will be part of the WorkPackage.
    // The authorizer will hash this payload and verify the signature against it.
    let payload_to_sign = b"my test payload for authorization".to_vec();
    let payload_hash = Sha256::digest(&payload_to_sign);

    // Sign the hash of the payload.
    let signature = signing_key.sign(&payload_hash);

    // Create the authorization credentials. We start with nonce = 0 for the first request.
    let creds = AuthCredentials {
        public_key: public_key_bytes,
        signature: signature.to_bytes(),
        nonce: 0,
    };

    // The fields for WorkItem have changed based on the compiler errors.
    let work_item = WorkItem {
        payload: WorkPayload::from(payload_to_sign),
        service: 1u32,
        code_hash: [0; 32].into(),
        refine_gas_limit: 1_000_000,
        accumulate_gas_limit: 500_000,
        import_segments: BoundedVec::default(),
        extrinsics: vec![],
        export_count: 0,
    };

    // The `items` field expects a `BoundedVec`, not a standard `Vec`.
    let items_vec = vec![work_item];
    let bounded_items: BoundedVec<WorkItem, MaxWorkItems> = BoundedVec::try_from(items_vec).expect(
        "Should not exceed max work items"
    );

    // RefineContext fields have changed.
    let refine_context = RefineContext {
        state_root: [0; 32].into(),
        anchor: Default::default(),
        beefy_root: [0; 32].into(),
        // `lookup_anchor` expects a HeaderHash, which can be defaulted.
        lookup_anchor: Default::default(),
        lookup_anchor_slot: 0,
        // `prerequisites` expects a VecSet, not a BoundedVec.
        prerequisites: VecSet::default(),
    };

    // The fields for WorkPackage have also changed.
    let work_package = WorkPackage {
        items: bounded_items,
        context: refine_context,
        // Authorizer is a struct that needs to be initialized with its fields.
        authorizer: Authorizer {
            code_hash: [0; 32].into(),
            param: Default::default(),
        },
        auth_code_host: 0,
        authorization: Authorization(creds.encode()),
    };

    // Define a core index.
    let core_index: CoreIndex = 1;

    // SCALE-encode the structures and then hex-encode the resulting bytes.
    let param_hex = hex::encode(creds.encode());
    let package_hex = hex::encode(work_package.encode());
    let core_index_hex = hex::encode(jam_codec::Encode::encode(&core_index));

    println!("param_hex: \"{}\"", param_hex);
    println!("package_hex: \"{}\"", package_hex);
    println!("core_index_hex: \"{}\"\n", core_index_hex);

    // --- 2. Data for /service/refine ---
    println!("## Endpoint: /service/refine");
    let service_id: ServiceId = 1;
    let payload = WorkPayload::from(b"Hello, JAM!".to_vec());
    let package_hash: Hash = [1; 32]; // Dummy hash

    // --- FIX START ---
    // Create a default RefineContext and encode it properly
    let default_refine_context = RefineContext {
        state_root: [0; 32].into(),
        anchor: Default::default(),
        beefy_root: [0; 32].into(),
        lookup_anchor: Default::default(),
        lookup_anchor_slot: 0,
        prerequisites: VecSet::default(),
    };
    let context_hex = hex::encode(default_refine_context.encode());
    // --- FIX END ---

    let auth_code_hash: Hash = [2; 32]; // Dummy hash

    println!("service_id_hex: \"{}\"", hex::encode(jam_codec::Encode::encode(&service_id)));
    println!("payload_hex: \"{}\"", hex::encode(payload.encode()));
    println!("package_hash_hex: \"{}\"", hex::encode(jam_codec::Encode::encode(&package_hash)));
    println!("context_hex: \"{}\"", context_hex);
    println!(
        "auth_code_hash_hex: \"{}\"\n",
        hex::encode(jam_codec::Encode::encode(&auth_code_hash))
    );

    // --- 3. Data for /service/accumulate ---
    println!("## Endpoint: /service/accumulate");
    let slot: Slot = 1;

    // AccumulateItem expects a PayloadHash, not a WorkPayload.
    let accumulate_payload = b"payload from successful work".to_vec();
    let accumulate_payload_hash = PayloadHash(Sha256::digest(&accumulate_payload).into());

    // The fields for AccumulateItem have changed.
    let accumulate_item = AccumulateItem {
        auth_output: AuthOutput(creds.encode()),
        payload: accumulate_payload_hash,
        result: Ok(WorkOutput::from(b"work output".to_vec())),
        package: WorkPackageHash([0; 32]),
        exports_root: [0; 32].into(),
        authorizer_hash: [0; 32].into(),
    };
    let items = vec![accumulate_item];

    println!("slot_hex: \"{}\"", hex::encode(jam_codec::Encode::encode(&slot)));
    println!("service_id_hex: \"{}\"", hex::encode(jam_codec::Encode::encode(&service_id)));
    println!("items_hex: \"{}\"\n", hex::encode(items.encode()));

    // --- 4. Data for /service/on_transfer ---
    println!("## Endpoint: /service/on_transfer");
    // Create a command to increment the counter.
    let command = ServiceCommand::IncrementCounter { by: 5 };

    // The Memo field expects a fixed-size array [u8; 128].
    // We encode the command, then copy it into the array.
    let encoded_command = command.encode();
    let mut memo_array = [0u8; 128];
    let len = encoded_command.len().min(128);
    memo_array[..len].copy_from_slice(&encoded_command[..len]);
    let memo = Memo(memo_array);

    // TransferRecord requires a gas_limit field.
    let transfer = TransferRecord {
        source: 123u32,
        destination: service_id.into(),
        amount: 1000u64,
        memo,
        gas_limit: 1_000_000,
    };

    let transfers = vec![transfer];

    println!("slot_hex: \"{}\"", hex::encode(jam_codec::Encode::encode(&slot)));
    println!("service_id_hex: \"{}\"", hex::encode(jam_codec::Encode::encode(&service_id)));
    println!("transfers_hex: \"{}\"\n", hex::encode(transfers.encode()));

    println!("--- End of Test Data ---");
}
