#ifndef BEM_OPERATOR_CONFIG_H
#define BEM_OPERATOR_CONFIG_H

#include <complex>
#include <cstring>

enum class BemSystemKind {
    PMCHWT,
    Balanced,
    Muller,
    MullerBalanced,
    Muller2,
    Muller2Balanced
};

struct BemOperatorConfig {
    BemSystemKind system = BemSystemKind::PMCHWT;
    std::complex<double> k_ext;
    std::complex<double> k_int;
    std::complex<double> eta_ext;
    std::complex<double> eta_int;
    int quad_order = 7;
};

struct BemBlockScales {
    double unknown_m_scale = 1.0;
    std::complex<double> row_h_scale = std::complex<double>(1.0, 0.0);
    double int_op_sign = 1.0;
    double k_identity = 0.0;
    bool n_form = false;
    double n_form_eps_int = 1.0;
    double n_form_m_identity = 0.0;
};

inline const char* bem_system_kind_name(BemSystemKind kind)
{
    switch (kind) {
    case BemSystemKind::PMCHWT: return "pmchwt";
    case BemSystemKind::Balanced: return "balanced";
    case BemSystemKind::Muller: return "muller";
    case BemSystemKind::MullerBalanced: return "muller-balanced";
    case BemSystemKind::Muller2: return "muller2";
    case BemSystemKind::Muller2Balanced: return "muller2-balanced";
    default: return "unknown";
    }
}

inline bool parse_bem_system_kind(const char* name, BemSystemKind& out)
{
    if (!name) return false;
    if (std::strcmp(name, "pmchwt") == 0) {
        out = BemSystemKind::PMCHWT;
    } else if (std::strcmp(name, "balanced") == 0) {
        out = BemSystemKind::Balanced;
    } else if (std::strcmp(name, "muller") == 0) {
        out = BemSystemKind::Muller;
    } else if (std::strcmp(name, "muller-balanced") == 0) {
        out = BemSystemKind::MullerBalanced;
    } else if (std::strcmp(name, "muller2") == 0) {
        out = BemSystemKind::Muller2;
    } else if (std::strcmp(name, "muller2-balanced") == 0) {
        out = BemSystemKind::Muller2Balanced;
    } else {
        return false;
    }
    return true;
}

inline BemSystemKind choose_default_bem_system(std::complex<double> refractive_index,
                                               bool auto_balanced_enabled,
                                               bool accurate_obj_profile)
{
    if (accurate_obj_profile)
        return BemSystemKind::Balanced;
    if (!auto_balanced_enabled)
        return BemSystemKind::PMCHWT;
    const double contrast = std::abs(refractive_index - std::complex<double>(1.0, 0.0));
    return (contrast >= 0.05) ? BemSystemKind::Balanced : BemSystemKind::PMCHWT;
}

inline BemSystemKind canonicalize_bem_system_kind(BemSystemKind requested,
                                                  bool experimental_nform_enabled)
{
    if (experimental_nform_enabled)
        return requested;
    if (requested == BemSystemKind::Muller2)
        return BemSystemKind::PMCHWT;
    if (requested == BemSystemKind::Muller2Balanced)
        return BemSystemKind::Balanced;
    return requested;
}

inline BemBlockScales bem_block_scales_for_system(BemSystemKind actual,
                                                  std::complex<double> refractive_index,
                                                  std::complex<double> eta_int,
                                                  bool experimental_nform_enabled)
{
    BemBlockScales scales;
    if (actual == BemSystemKind::Muller || actual == BemSystemKind::MullerBalanced ||
        experimental_nform_enabled)
        scales.int_op_sign = -1.0;

    if (actual == BemSystemKind::Balanced || actual == BemSystemKind::MullerBalanced ||
        actual == BemSystemKind::Muller2Balanced) {
        scales.unknown_m_scale = std::abs(refractive_index);
        scales.row_h_scale = eta_int;
    }

    if (experimental_nform_enabled) {
        scales.n_form = true;
        scales.k_identity = -1.0;
        scales.n_form_eps_int = std::norm(refractive_index);
        scales.n_form_m_identity = -0.5 * (1.0 + scales.n_form_eps_int);
        scales.unknown_m_scale = 1.0;
        scales.row_h_scale = std::complex<double>(1.0, 0.0);
    }
    return scales;
}

inline void form_pmchwt_matrix(
    int N,
    const std::complex<double>* L_ext,
    const std::complex<double>* K_ext,
    const std::complex<double>* L_int,
    const std::complex<double>* K_int,
    std::complex<double> eta_ext,
    std::complex<double> eta_int,
    std::complex<double>* Z)
{
    const int N2 = 2 * N;
    const std::complex<double> inv_eta_ext = 1.0 / eta_ext;
    const std::complex<double> inv_eta_int = 1.0 / eta_int;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            const int ij = i * N + j;
            const std::complex<double> le = L_ext[ij];
            const std::complex<double> li = L_int[ij];
            const std::complex<double> ksum = K_ext[ij] + K_int[ij];

            Z[i * N2 + j] = eta_ext * le + eta_int * li;
            Z[i * N2 + (N + j)] = -ksum;
            Z[(N + i) * N2 + j] = ksum;
            Z[(N + i) * N2 + (N + j)] = inv_eta_ext * le + inv_eta_int * li;
        }
    }
}

inline void form_bem_system_matrix(
    int N,
    const std::complex<double>* L_ext,
    const std::complex<double>* K_ext,
    const std::complex<double>* L_int,
    const std::complex<double>* K_int,
    std::complex<double> eta_ext,
    std::complex<double> eta_int,
    double unknown_m_scale,
    std::complex<double> row_h_scale,
    double int_op_sign,
    double k_identity,
    std::complex<double>* Z)
{
    const int N2 = 2 * N;
    const std::complex<double> inv_eta_ext = 1.0 / eta_ext;
    const std::complex<double> inv_eta_int = 1.0 / eta_int;
    const double inv_m_scale = 1.0 / unknown_m_scale;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            const int ij = i * N + j;
            const std::complex<double> le = L_ext[ij];
            const std::complex<double> li = L_int[ij];
            std::complex<double> ksum = K_ext[ij] + int_op_sign * K_int[ij];
            if (i == j)
                ksum += k_identity;

            Z[i * N2 + j] = eta_ext * le + int_op_sign * eta_int * li;
            Z[i * N2 + (N + j)] = -ksum * inv_m_scale;
            Z[(N + i) * N2 + j] = row_h_scale * ksum;
            Z[(N + i) * N2 + (N + j)] =
                row_h_scale * (inv_eta_ext * le + int_op_sign * inv_eta_int * li) * inv_m_scale;
        }
    }
}

inline void form_bem_system_matrix(
    int N,
    const std::complex<double>* L_ext,
    const std::complex<double>* K_ext,
    const std::complex<double>* L_int,
    const std::complex<double>* K_int,
    std::complex<double> eta_ext,
    std::complex<double> eta_int,
    const BemBlockScales& scales,
    std::complex<double>* Z)
{
    form_bem_system_matrix(N, L_ext, K_ext, L_int, K_int, eta_ext, eta_int,
                           scales.unknown_m_scale, scales.row_h_scale,
                           scales.int_op_sign, scales.k_identity, Z);
}

#endif // BEM_OPERATOR_CONFIG_H
