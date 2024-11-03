#![allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::unwrap_used
)]

use crate::mock::*;
use frame_support::{assert_err, assert_ok};
use frame_system::Config;
use pallet_subtensor::epoch::math::safe_exp;
use pallet_subtensor::*;
use rand::{distributions::Uniform, rngs::StdRng, seq::SliceRandom, thread_rng, Rng, SeedableRng};
use sp_core::U256;
use sp_runtime::DispatchError;
use std::time::Instant;
use substrate_fixed::types::I32F32;

mod mock;

pub fn fixed(val: f32) -> I32F32 {
    I32F32::from_num(val)
}

pub fn fixed_to_u16(x: I32F32) -> u16 {
    x.to_num::<u16>()
}

pub fn fixed_proportion_to_u16(x: I32F32) -> u16 {
    fixed_to_u16(x * I32F32::from_num(u16::MAX))
}

// Normalizes (sum to 1 except 0) the input vector directly in-place.
#[allow(dead_code)]
pub fn inplace_normalize(x: &mut [I32F32]) {
    let x_sum: I32F32 = x.iter().sum();
    if x_sum == I32F32::from_num(0.0_f32) {
        return;
    }
    for i in x.iter_mut() {
        *i /= x_sum;
    }
}

// Inplace normalize the passed positive integer weights so that they sum to u16 max value.
fn normalize_weights(mut weights: Vec<u16>) -> Vec<u16> {
    let sum: u64 = weights.iter().map(|x| *x as u64).sum();
    if sum == 0 {
        return weights;
    }
    weights.iter_mut().for_each(|x| {
        *x = (*x as u64 * u16::MAX as u64 / sum) as u16;
    });
    weights
}

// // Return as usize an I32F32 ratio of a usize input, avoiding the 0% and 100% extremes.
// fn non_extreme_fixed_ratio(ratio: I32F32, total: usize) -> usize {
//     if total == 0 {
//         return total;
//     }
//     let mut subset: usize = (ratio * I32F32::from_num(total)).to_num::<usize>();
//     if subset == 0 {
//         subset = 1;
//     } else if subset == total {
//         subset = total - 1;
//     }
//     return subset;
// }

// // Box-Muller Transform converting two uniform random samples to a normal random sample.
// fn normal(size: usize, rng: &mut StdRng, dist: &Uniform<u16>) -> Vec<I32F32> {
//     let max: I32F32 = I32F32::from_num(u16::MAX);
//     let two: I32F32 = I32F32::from_num(2);
//     let eps: I32F32 = I32F32::from_num(0.000001);
//     let pi: I32F32 = I32F32::from_num(PI);

//     let uniform_u16: Vec<u16> = (0..(2 * size)).map(|_| rng.sample(&dist)).collect();
//     let uniform: Vec<I32F32> = uniform_u16
//         .iter()
//         .map(|&x| I32F32::from_num(x) / max)
//         .collect();
//     let mut normal: Vec<I32F32> = vec![I32F32::from_num(0); size as usize];

//     for i in 0..size {
//         let u1: I32F32 = uniform[i] + eps;
//         let u2: I32F32 = uniform[i + size] + eps;
//         normal[i] = sqrt::<I32F32, I32F32>(-two * ln::<I32F32, I32F32>(u1).expect("")).expect("")
//             * cos(two * pi * u2);
//     }
//     normal
// }

// Returns validators and servers uids with either blockwise, regular, or random interleaving.
fn distribute_nodes(
    validators_n: usize,
    network_n: usize,
    interleave: usize,
) -> (Vec<u16>, Vec<u16>) {
    let mut validators: Vec<u16> = vec![];
    let mut servers: Vec<u16> = vec![];

    if interleave == 0 {
        // blockwise [validator_block, server_block]
        validators = (0..validators_n as u16).collect();
        servers = (validators_n as u16..network_n as u16).collect();
    } else if interleave == 1 {
        // regular interleaving [val, srv, srv, ..., srv, val, srv, srv, ..., srv, val, srv, ..., srv]
        (validators, servers) = (0..network_n as u16)
            .collect::<Vec<u16>>()
            .iter()
            .partition(|&i| *i as usize % (network_n / validators_n) == 0);
    } else if interleave == 2 {
        // random interleaving
        let mut permuted_uids: Vec<u16> = (0..network_n as u16).collect();
        permuted_uids.shuffle(&mut thread_rng());
        validators = permuted_uids[0..validators_n].into();
        servers = permuted_uids[validators_n..network_n].into();
    }

    (validators, servers)
}

#[allow(dead_code)]
fn uid_stats(netuid: u16, uid: u16) {
    log::info!(
        "stake: {:?}",
        SubtensorModule::get_total_stake_for_hotkey(&(U256::from(uid)))
    );
    log::info!("rank: {:?}", SubtensorModule::get_rank_for_uid(netuid, uid));
    log::info!(
        "trust: {:?}",
        SubtensorModule::get_trust_for_uid(netuid, uid)
    );
    log::info!(
        "consensus: {:?}",
        SubtensorModule::get_consensus_for_uid(netuid, uid)
    );
    log::info!(
        "incentive: {:?}",
        SubtensorModule::get_incentive_for_uid(netuid, uid)
    );
    log::info!(
        "dividend: {:?}",
        SubtensorModule::get_dividends_for_uid(netuid, uid)
    );
    log::info!(
        "emission: {:?}",
        SubtensorModule::get_emission_for_uid(netuid, uid)
    );
}

#[allow(clippy::too_many_arguments)]
fn init_run_epochs(
    netuid: u16,
    n: u16,
    validators: &[u16],
    servers: &[u16],
    epochs: u16,
    stake_per_validator: u64,
    server_self: bool,
    input_stake: &[u64],
    use_input_stake: bool,
    input_weights: &[Vec<(u16, u16)>],
    use_input_weights: bool,
    random_weights: bool,
    random_seed: u64,
    sparse: bool,
) {
    // === Create the network
    add_network(netuid, u16::MAX - 1, 0); // set higher tempo to avoid built-in epoch, then manual epoch instead

    // === Register uids
    SubtensorModule::set_max_allowed_uids(netuid, n);
    for key in 0..n {
        let stake = if use_input_stake {
            input_stake[key as usize]
        } else if validators.contains(&key) {
            stake_per_validator
        } else {
            // only validators receive stake
            0
        };

        // let stake: u64 = 1; // alternative test: all nodes receive stake, should be same outcome, except stake
        SubtensorModule::add_balance_to_coldkey_account(&(U256::from(key)), stake);
        SubtensorModule::append_neuron(netuid, &(U256::from(key)), 0);
        SubtensorModule::increase_stake_on_coldkey_hotkey_account(
            &U256::from(key),
            &U256::from(key),
            stake,
        );
    }
    assert_eq!(SubtensorModule::get_subnetwork_n(netuid), n);

    // === Issue validator permits
    SubtensorModule::set_max_allowed_validators(netuid, validators.len() as u16);
    assert_eq!(
        SubtensorModule::get_max_allowed_validators(netuid),
        validators.len() as u16
    );
    SubtensorModule::epoch(netuid, 1_000_000_000); // run first epoch to set allowed validators
    run_to_block(1); // run to next block to ensure weights are set on nodes after their registration block

    // === Set weights
    let mut rng = StdRng::seed_from_u64(random_seed); // constant seed so weights over multiple runs are equal
    let range = Uniform::new(0, u16::MAX);
    let mut weights: Vec<u16> = vec![u16::MAX / n; servers.len()];
    for uid in validators {
        if random_weights {
            weights = (0..servers.len()).map(|_| rng.sample(range)).collect();
            weights = normalize_weights(weights);
            // assert_eq!(weights.iter().map(|x| *x as u64).sum::<u64>(), u16::MAX as u64); // normalized weight sum not always u16::MAX
        }
        if use_input_weights {
            let sparse_weights = input_weights[*uid as usize].clone();
            weights = sparse_weights.iter().map(|(_, w)| *w).collect();
            let srvs: Vec<u16> = sparse_weights.iter().map(|(s, _)| *s).collect();
            assert_ok!(SubtensorModule::set_weights(
                RuntimeOrigin::signed(U256::from(*uid as u64)),
                netuid,
                srvs,
                weights.clone(),
                0
            ));
        } else {
            assert_ok!(SubtensorModule::set_weights(
                RuntimeOrigin::signed(U256::from(*uid as u64)),
                netuid,
                servers.to_vec(),
                weights.clone(),
                0
            ));
        }
    }
    if server_self {
        for uid in servers {
            assert_ok!(SubtensorModule::set_weights(
                RuntimeOrigin::signed(U256::from(*uid as u64)),
                netuid,
                vec![*uid],
                vec![u16::MAX],
                0
            )); // server self-weight
        }
    }

    // === Run the epochs.
    log::info!("Start {epochs} epoch(s)");
    let start = Instant::now();
    for _ in 0..epochs {
        if sparse {
            SubtensorModule::epoch(netuid, 1_000_000_000);
        } else {
            SubtensorModule::epoch_dense(netuid, 1_000_000_000);
        }
    }
    let duration = start.elapsed();
    log::info!(
        "Time elapsed in (sparse={sparse}) epoch() is: {:?}",
        duration
    );

    // let bonds = SubtensorModule::get_bonds( netuid );
    // for (uid, node) in vec![ (validators[0], "validator"), (servers[0], "server") ] {
    // 	log::info!("\n{node}" );
    // 	uid_stats(netuid, uid);
    // 	log::info!("bonds: {:?} (on validator), {:?} (on server)", bonds[uid as usize][0], bonds[uid as usize][servers[0] as usize]);
    // }
}

// // Generate a random graph that is split into a major and minor set, each setting specific weight on itself and the complement on the other.
// fn split_graph(
//     major_stake: I32F32,
//     major_weight: I32F32,
//     minor_weight: I32F32,
//     weight_stddev: I32F32,
//     validators_n: usize,
//     network_n: usize,
//     interleave: usize,
// ) -> (
//     Vec<u16>,
//     Vec<u16>,
//     Vec<u16>,
//     Vec<u16>,
//     Vec<u16>,
//     Vec<u16>,
//     Vec<u64>,
//     Vec<Vec<(u16, u16)>>,
//     I32F32,
// ) {
//     let servers_n: usize = network_n - validators_n;
//     let major_servers_n: usize = non_extreme_fixed_ratio(major_stake, servers_n);
//     let major_validators_n: usize = non_extreme_fixed_ratio(major_stake, validators_n);

//     let (validators, servers) = distribute_nodes(validators_n, network_n, interleave as usize);
//     let major_validators: Vec<u16> = (0..major_validators_n).map(|i| validators[i]).collect();
//     let minor_validators: Vec<u16> = (major_validators_n..validators_n)
//         .map(|i| validators[i])
//         .collect();
//     let major_servers: Vec<u16> = (0..major_servers_n).map(|i| servers[i]).collect();
//     let minor_servers: Vec<u16> = (major_servers_n..servers_n).map(|i| servers[i]).collect();

//     let zero: I32F32 = I32F32::from_num(0);
//     let one: I32F32 = I32F32::from_num(1);
//     let stddev: I32F32 = I32F32::from_num(0.3);
//     let total_stake: I64F64 = I64F64::from_num(21_000_000_000_000_000 as u64);
//     let mut rng = StdRng::seed_from_u64(0); // constant seed so weights over multiple runs are equal
//     let dist = Uniform::new(0, u16::MAX);

//     let mut stake: Vec<u64> = vec![0; network_n];
//     let mut stake_fixed: Vec<I32F32> = vec![zero; network_n];
//     for (ratio, vals) in vec![
//         (major_stake, &major_validators),
//         (one - major_stake, &minor_validators),
//     ] {
//         let mut sample = normal(vals.len(), &mut rng, &dist)
//             .iter()
//             .map(|x: &I32F32| {
//                 let v: I32F32 = (stddev * x) + one;
//                 if v < zero {
//                     zero
//                 } else {
//                     v
//                 }
//             })
//             .collect();
//         inplace_normalize(&mut sample);
//         for (i, &val) in vals.iter().enumerate() {
//             stake[val as usize] =
//                 (I64F64::from_num(ratio) * I64F64::from_num(sample[i]) * total_stake)
//                     .to_num::<u64>();
//             stake_fixed[val as usize] =
//                 I32F32::from_num(I64F64::from_num(ratio) * I64F64::from_num(sample[i]));
//         }
//     }

//     let mut weights: Vec<Vec<(u16, u16)>> = vec![vec![]; network_n as usize];
//     let mut weights_fixed: Vec<Vec<I32F32>> = vec![vec![zero; network_n]; network_n];
//     for (first, second, vals) in vec![
//         (major_weight, one - major_weight, &major_validators),
//         (one - minor_weight, minor_weight, &minor_validators),
//     ] {
//         for &val in vals {
//             for (weight, srvs) in vec![(first, &major_servers), (second, &minor_servers)] {
//                 let mut sample: Vec<I32F32> = normal(srvs.len(), &mut rng, &dist)
//                     .iter()
//                     .map(|x: &I32F32| {
//                         let v: I32F32 = (weight_stddev * x) + one;
//                         if v < zero {
//                             zero
//                         } else {
//                             v
//                         }
//                     })
//                     .collect();
//                 inplace_normalize(&mut sample);

//                 for (i, &srv) in srvs.iter().enumerate() {
//                     weights[val as usize].push((srv, fixed_proportion_to_u16(weight * sample[i])));
//                     weights_fixed[val as usize][srv as usize] = weight * sample[i];
//                 }
//             }
//             inplace_normalize(&mut weights_fixed[val as usize]);
//         }
//     }

//     inplace_normalize(&mut stake_fixed);

//     // Calculate stake-weighted mean per server
//     let mut weight_mean: Vec<I32F32> = vec![zero; network_n];
//     for val in 0..network_n {
//         if stake_fixed[val] > zero {
//             for srv in 0..network_n {
//                 weight_mean[srv] += stake_fixed[val] * weights_fixed[val][srv];
//             }
//         }
//     }

//     // Calculate stake-weighted absolute standard deviation
//     let mut weight_dev: Vec<I32F32> = vec![zero; network_n];
//     for val in 0..network_n {
//         if stake_fixed[val] > zero {
//             for srv in 0..network_n {
//                 weight_dev[srv] +=
//                     stake_fixed[val] * (weight_mean[srv] - weights_fixed[val][srv]).abs();
//             }
//         }
//     }

//     // Calculate rank-weighted mean of weight_dev
//     let avg_weight_dev: I32F32 =
//         weight_dev.iter().sum::<I32F32>() / weight_mean.iter().sum::<I32F32>();

//     (
//         validators,
//         servers,
//         major_validators,
//         minor_validators,
//         major_servers,
//         minor_servers,
//         stake,
//         weights,
//         avg_weight_dev,
//     )
// }

// Test consensus guarantees with an epoch on a graph with 4096 nodes, of which the first 128 are validators, the graph is split into a major and minor set, each setting specific weight on itself and the complement on the other. Asserts that the major emission ratio >= major stake ratio.
// #[test]
// fn test_consensus_guarantees() {
//     let netuid: u16 = 0;
//     let network_n: u16 = 512;
//     let validators_n: u16 = 64;
//     let epochs: u16 = 1;
//     let interleave = 2;
//     log::info!("test_consensus_guarantees ({network_n:?}, {validators_n:?} validators)");
//     for (major_stake, major_weight, minor_weight, weight_stddev) in vec![
//         (0.51, 1., 1., 0.001),
//         (0.51, 0.03, 0., 0.001),
//         (0.51, 0.51, 0.49, 0.001),
//         (0.51, 0.51, 1., 0.001),
//         (0.51, 0.61, 0.8, 0.1),
//         (0.6, 0.67, 0.65, 0.2),
//         (0.6, 0.74, 0.77, 0.4),
//         (0.6, 0.76, 0.8, 0.4),
//         (0.6, 0.76, 1., 0.4),
//         (0.6, 0.92, 1., 0.4),
//         (0.6, 0.94, 1., 0.4),
//         (0.65, 0.78, 0.85, 0.6),
//         (0.7, 0.81, 0.85, 0.8),
//         (0.7, 0.83, 0.85, 1.),
//     ] {
//         let (
//             validators,
//             servers,
//             major_validators,
//             minor_validators,
//             major_servers,
//             minor_servers,
//             stake,
//             weights,
//             _avg_weight_dev,
//         ) = split_graph(
//             fixed(major_stake),
//             fixed(major_weight),
//             fixed(minor_weight),
//             fixed(weight_stddev),
//             validators_n as usize,
//             network_n as usize,
//             interleave as usize,
//         );

//         new_test_ext(1).execute_with(|| {
//             init_run_epochs(
//                 netuid,
//                 network_n,
//                 &validators,
//                 &servers,
//                 epochs,
//                 1,
//                 true,
//                 &stake,
//                 true,
//                 &weights,
//                 true,
//                 false,
//                 0,
//                 false,
//             );

//             let mut major_emission: I64F64 = I64F64::from_num(0);
//             let mut minor_emission: I64F64 = I64F64::from_num(0);
//             for set in vec![major_validators, major_servers] {
//                 for uid in set {
//                     major_emission +=
//                         I64F64::from_num(SubtensorModule::get_emission_for_uid(netuid, uid));
//                 }
//             }
//             for set in vec![minor_validators, minor_servers] {
//                 for uid in set {
//                     minor_emission +=
//                         I64F64::from_num(SubtensorModule::get_emission_for_uid(netuid, uid));
//                 }
//             }
//             let major_ratio: I32F32 =
//                 I32F32::from_num(major_emission / (major_emission + minor_emission));
//             assert!(major_stake <= major_ratio);
//         });
//     }
// }

// Test an epoch on an empty graph.
// #[test]
// fn test_overflow() {
//     new_test_ext(1).execute_with(|| {
//         log::info!("test_overflow:");
//         let netuid: u16 = 1;
//         add_network(netuid, 0, 0);
//         SubtensorModule::set_max_allowed_uids(netuid, 3);
//         SubtensorModule::increase_stake_on_coldkey_hotkey_account(
//             &U256::from(0),
//             &U256::from(0),
//             10,
//         );
//         SubtensorModule::increase_stake_on_coldkey_hotkey_account(
//             &U256::from(1),
//             &U256::from(1),
//             10,
//         );
//         SubtensorModule::increase_stake_on_coldkey_hotkey_account(
//             &U256::from(2),
//             &U256::from(2),
//             10,
//         );
//         SubtensorModule::append_neuron(netuid, &U256::from(0), 0);
//         SubtensorModule::append_neuron(netuid, &U256::from(1), 0);
//         SubtensorModule::append_neuron(netuid, &U256::from(2), 0);
//         SubtensorModule::set_validator_permit_for_uid(0, 0, true);
//         SubtensorModule::set_validator_permit_for_uid(0, 1, true);
//         SubtensorModule::set_validator_permit_for_uid(0, 2, true);
//         assert_ok!(SubtensorModule::set_weights(
//             RuntimeOrigin::signed(U256::from(0)),
//             netuid,
//             vec![0, 1, 2],
//             vec![u16::MAX / 3, u16::MAX / 3, u16::MAX],
//             0
//         ));
//         assert_ok!(SubtensorModule::set_weights(
//             RuntimeOrigin::signed(U256::from(1)),
//             netuid,
//             vec![1, 2],
//             vec![u16::MAX / 2, u16::MAX / 2],
//             0
//         ));
//         assert_ok!(SubtensorModule::set_weights(
//             RuntimeOrigin::signed(U256::from(2)),
//             netuid,
//             vec![2],
//             vec![u16::MAX],
//             0
//         ));
//         SubtensorModule::epoch(0, u64::MAX);
//     });
// }

// Test an epoch on an empty graph.
// #[test]
// fn test_nill_epoch_subtensor() {
//     new_test_ext(1).execute_with(|| {
//         log::info!("test_nill_epoch:");
//         SubtensorModule::epoch(0, 0);
//     });
// }

// // Map the retention graph for consensus guarantees with an single epoch on a graph with 512 nodes, of which the first 64 are validators, the graph is split into a major and minor set, each setting specific weight on itself and the complement on the other.
// //
// // ```import torch
// // import matplotlib.pyplot as plt
// // from matplotlib.pyplot import cm
// // %matplotlib inline
// //
// // with open('finney_consensus_0.4.txt') as f:  # test output saved to finney_consensus.txt
// //     retention_map = eval(f.read())
// //
// // major_ratios = {}
// // avg_weight_devs = {}
// // for major_stake, major_weight, minor_weight, avg_weight_dev, major_ratio in retention_map:
// //     major_stake = f'{major_stake:.2f}'
// //     maj, min = int(round(50 * major_weight)), int(round(50 * minor_weight))
// //     avg_weight_devs.setdefault(major_stake, torch.zeros((51, 51)))
// //     avg_weight_devs[major_stake][maj][min] = avg_weight_dev
// //     major_ratios.setdefault(major_stake, torch.zeros((51, 51)))
// //     major_ratios[major_stake][maj][min] = major_ratio
// //
// // _x = torch.linspace(0, 1, 51); _y = torch.linspace(0, 1, 51)
// // x, y = torch.meshgrid(_x, _y, indexing='ij')
// //
// // fig = plt.figure(figsize=(6, 6), dpi=70); ax = fig.gca()
// // ax.set_xticks(torch.arange(0, 1, 0.05)); ax.set_yticks(torch.arange(0, 1., 0.05))
// // ax.set_xticklabels([f'{_:.2f}'[1:] for _ in torch.arange(0, 1., 0.05)])
// // plt.grid(); plt.rc('grid', linestyle="dotted", color=[0.85, 0.85, 0.85])
// //
// // isolate = ['0.60']; stakes = [0.51, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
// // colors = cm.viridis(torch.linspace(0, 1, len(stakes) + 1))
// // for i, stake in enumerate(stakes):
// //     contours = plt.contour(x, y, major_ratios[f'{stake:.2f}'], levels=[0., stake], colors=[colors[i + 1]])
// //     if f'{stake:.2f}' in isolate:
// //         contours.collections[1].set_linewidth(3)
// //     plt.clabel(contours, inline=True, fontsize=10)
// //
// // plt.title(f'Major emission [$stake_{{maj}}=emission_{{maj}}$ retention lines]')
// // plt.ylabel('Minor self-weight'); plt.xlabel('Major self-weight'); plt.show()
// // ```
// // #[test]
// fn _map_consensus_guarantees() {
//     let netuid: u16 = 1;
//     let network_n: u16 = 512;
//     let validators_n: u16 = 64;
//     let epochs: u16 = 1;
//     let interleave = 0;
//     let weight_stddev: I32F32 = fixed(0.4);
//     println!("[");
//     for _major_stake in vec![0.51, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99] {
//         let major_stake: I32F32 = I32F32::from_num(_major_stake);
//         for _major_weight in 0..51 {
//             let major_weight: I32F32 = I32F32::from_num(50 - _major_weight) / I32F32::from_num(50);
//             for _minor_weight in 0..51 {
//                 let minor_weight: I32F32 =
//                     I32F32::from_num(50 - _minor_weight) / I32F32::from_num(50);
//                 let (
//                     validators,
//                     servers,
//                     major_validators,
//                     minor_validators,
//                     major_servers,
//                     minor_servers,
//                     stake,
//                     weights,
//                     avg_weight_dev,
//                 ) = split_graph(
//                     major_stake,
//                     major_weight,
//                     minor_weight,
//                     weight_stddev,
//                     validators_n as usize,
//                     network_n as usize,
//                     interleave as usize,
//                 );

//                 new_test_ext(1).execute_with(|| {
// 					init_run_epochs(netuid, network_n, &validators, &servers, epochs, 1, true, &stake, true, &weights, true, false, 0, true);

// 					let mut major_emission: I64F64 = I64F64::from_num(0);
// 					let mut minor_emission: I64F64 = I64F64::from_num(0);
// 					for set in vec![major_validators, major_servers] {
// 						for uid in set {
// 							major_emission += I64F64::from_num(SubtensorModule::get_emission_for_uid( netuid, uid ));
// 						}
// 					}
// 					for set in vec![minor_validators, minor_servers] {
// 						for uid in set {
// 							minor_emission += I64F64::from_num(SubtensorModule::get_emission_for_uid( netuid, uid ));
// 						}
// 					}
// 					let major_ratio: I32F32 = I32F32::from_num(major_emission / (major_emission + minor_emission));
// 					println!("[{major_stake}, {major_weight:.2}, {minor_weight:.2}, {avg_weight_dev:.3}, {major_ratio:.3}], ");
// 				});
//             }
//         }
//     }
//     println!("]");
// }
