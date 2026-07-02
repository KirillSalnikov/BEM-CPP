#include "../src/operator_config.h"
#include "../src/gpu_select.h"

#include <cassert>
#include <complex>
#include <cmath>
#include <cstdlib>

static bool close_complex(std::complex<double> a, std::complex<double> b)
{
    return std::abs(a - b) < 1e-12;
}

static void require_system_matrix_matches_scales(
    const std::complex<double>* le,
    const std::complex<double>* ke,
    const std::complex<double>* li,
    const std::complex<double>* ki,
    std::complex<double> eta_e,
    std::complex<double> eta_i,
    const BemBlockScales& scales)
{
    const int n = 2;
    std::complex<double> z[16];
    form_bem_system_matrix(n, le, ke, li, ki, eta_e, eta_i, scales, z);
    const double inv_s = 1.0 / scales.unknown_m_scale;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            const int ij = i * n + j;
            std::complex<double> ksum = ke[ij] + scales.int_op_sign * ki[ij];
            if (i == j)
                ksum += std::complex<double>(scales.k_identity, 0.0);
            assert(close_complex(z[i * 2 * n + j],
                                 eta_e * le[ij] + scales.int_op_sign * eta_i * li[ij]));
            assert(close_complex(z[i * 2 * n + (n + j)], -inv_s * ksum));
            assert(close_complex(z[(n + i) * 2 * n + j], scales.row_h_scale * ksum));
            assert(close_complex(z[(n + i) * 2 * n + (n + j)],
                                 scales.row_h_scale * inv_s *
                                     (le[ij] / eta_e + scales.int_op_sign * li[ij] / eta_i)));
        }
    }
}

int main()
{
    BemSystemKind kind;
    assert(parse_bem_system_kind("pmchwt", kind));
    assert(kind == BemSystemKind::PMCHWT);
    assert(parse_bem_system_kind("muller2-balanced", kind));
    assert(kind == BemSystemKind::Muller2Balanced);
    assert(!parse_bem_system_kind("bad-system", kind));
    assert(choose_default_bem_system({1.3116, 0.0}, true, false) == BemSystemKind::Balanced);
    assert(choose_default_bem_system({1.01, 0.0}, true, false) == BemSystemKind::PMCHWT);
    assert(choose_default_bem_system({1.3116, 0.0}, false, false) == BemSystemKind::PMCHWT);
    assert(choose_default_bem_system({1.0, 0.0}, false, true) == BemSystemKind::Balanced);
    assert(canonicalize_bem_system_kind(BemSystemKind::Muller2, false) == BemSystemKind::PMCHWT);
    assert(canonicalize_bem_system_kind(BemSystemKind::Muller2Balanced, false) == BemSystemKind::Balanced);
    assert(canonicalize_bem_system_kind(BemSystemKind::Muller2Balanced, true) == BemSystemKind::Muller2Balanced);
    assert(canonicalize_bem_system_kind(BemSystemKind::MullerBalanced, false) == BemSystemKind::MullerBalanced);

    const std::complex<double> m(1.6, 0.2);
    const std::complex<double> eta_m = 1.0 / m;
    BemBlockScales pmchwt_scales = bem_block_scales_for_system(BemSystemKind::PMCHWT, m, eta_m, false);
    assert(pmchwt_scales.unknown_m_scale == 1.0);
    assert(close_complex(pmchwt_scales.row_h_scale, {1.0, 0.0}));
    assert(pmchwt_scales.int_op_sign == 1.0);
    assert(pmchwt_scales.k_identity == 0.0);
    assert(!pmchwt_scales.n_form);

    BemBlockScales balanced_scales = bem_block_scales_for_system(BemSystemKind::Balanced, m, eta_m, false);
    assert(std::abs(balanced_scales.unknown_m_scale - std::abs(m)) < 1e-12);
    assert(close_complex(balanced_scales.row_h_scale, eta_m));
    assert(balanced_scales.int_op_sign == 1.0);
    assert(!balanced_scales.n_form);

    BemBlockScales muller_scales = bem_block_scales_for_system(BemSystemKind::Muller, m, eta_m, false);
    assert(muller_scales.unknown_m_scale == 1.0);
    assert(close_complex(muller_scales.row_h_scale, {1.0, 0.0}));
    assert(muller_scales.int_op_sign == -1.0);
    assert(!muller_scales.n_form);

    BemBlockScales nform_scales = bem_block_scales_for_system(BemSystemKind::Muller2Balanced, m, eta_m, true);
    assert(nform_scales.unknown_m_scale == 1.0);
    assert(close_complex(nform_scales.row_h_scale, {1.0, 0.0}));
    assert(nform_scales.int_op_sign == -1.0);
    assert(nform_scales.k_identity == -1.0);
    assert(nform_scales.n_form);
    assert(std::abs(nform_scales.n_form_eps_int - std::norm(m)) < 1e-12);
    assert(std::abs(nform_scales.n_form_m_identity + 0.5 * (1.0 + std::norm(m))) < 1e-12);

    std::vector<int> gpu_list = bem_parse_gpu_list_env("2, 4");
    assert(gpu_list.size() == 2);
    assert(gpu_list[0] == 2);
    assert(gpu_list[1] == 4);
    assert(bem_validate_gpu_list(gpu_list, 5));
    assert(!bem_validate_gpu_list(gpu_list, 4));
    assert(!bem_validate_gpu_list(bem_parse_gpu_list_env("1,1"), 4));
    assert(bem_parse_gpu_list_env("bad").empty());
    assert(!bem_env_value_enabled(nullptr));
    assert(bem_env_value_enabled(""));
    assert(bem_env_value_enabled("1"));
    assert(bem_env_value_enabled(" true "));
    assert(bem_env_value_enabled("YES"));
    assert(!bem_env_value_enabled("0"));
    assert(!bem_env_value_enabled(" false "));
    assert(!bem_env_value_enabled("OFF"));
    assert(bem_env_value_enabled("unexpected"));
    unsetenv("BEM_TEST_FLAG_PRESENT");
    assert(!bem_env_flag_present("BEM_TEST_FLAG_PRESENT"));
    assert(!bem_env_has_value("BEM_TEST_FLAG_PRESENT"));
    setenv("BEM_TEST_FLAG_PRESENT", "0", 1);
    assert(bem_env_flag_present("BEM_TEST_FLAG_PRESENT"));
    assert(bem_env_has_value("BEM_TEST_FLAG_PRESENT"));
    assert(!bem_env_flag_enabled("BEM_TEST_FLAG_PRESENT"));
    setenv("BEM_TEST_FLAG_PRESENT", "   ", 1);
    assert(bem_env_flag_present("BEM_TEST_FLAG_PRESENT"));
    assert(!bem_env_has_value("BEM_TEST_FLAG_PRESENT"));
    setenv("BEM_TEST_FLAG_PRESENT", "yes", 1);
    assert(bem_env_flag_enabled("BEM_TEST_FLAG_PRESENT"));
    assert(bem_env_has_value("BEM_TEST_FLAG_PRESENT"));
    unsetenv("BEM_TEST_FLAG_PRESENT");
    assert(bem_env_int("BEM_TEST_NUMERIC", 17) == 17);
    setenv("BEM_TEST_NUMERIC", "42", 1);
    assert(bem_env_int("BEM_TEST_NUMERIC", 17) == 42);
    setenv("BEM_TEST_NUMERIC", "  -3 ", 1);
    assert(bem_env_int("BEM_TEST_NUMERIC", 17) == -3);
    setenv("BEM_TEST_NUMERIC", "12bad", 1);
    assert(bem_env_int("BEM_TEST_NUMERIC", 17) == 17);
    setenv("BEM_TEST_NUMERIC", "   ", 1);
    assert(bem_env_int("BEM_TEST_NUMERIC", 17) == 17);
    setenv("BEM_TEST_NUMERIC", "2.5", 1);
    assert(std::abs(bem_env_double("BEM_TEST_NUMERIC", 1.0) - 2.5) < 1e-12);
    setenv("BEM_TEST_NUMERIC", "nan", 1);
    assert(bem_env_double("BEM_TEST_NUMERIC", 1.0) == 1.0);
    unsetenv("BEM_TEST_NUMERIC");

    std::complex<double> le[4] = {
        {1.0, 0.5}, {2.0, -0.25},
        {-1.0, 0.75}, {0.5, 0.0},
    };
    std::complex<double> ke[4] = {
        {0.2, 0.1}, {-0.3, 0.4},
        {0.7, -0.2}, {0.0, 0.5},
    };
    std::complex<double> li[4] = {
        {0.5, -0.1}, {-0.4, 0.2},
        {0.1, 0.3}, {1.5, -0.7},
    };
    std::complex<double> ki[4] = {
        {-0.2, 0.0}, {0.1, -0.4},
        {-0.7, 0.2}, {0.3, -0.5},
    };
    std::complex<double> z[16];
    const std::complex<double> eta_e(1.0, 0.0);
    const std::complex<double> eta_i(0.7, 0.15);
    form_pmchwt_matrix(2, le, ke, li, ki, eta_e, eta_i, z);

    const int n = 2;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            const int ij = i * n + j;
            const std::complex<double> ksum = ke[ij] + ki[ij];
            assert(close_complex(z[i * 2 * n + j], eta_e * le[ij] + eta_i * li[ij]));
            assert(close_complex(z[i * 2 * n + (n + j)], -ksum));
            assert(close_complex(z[(n + i) * 2 * n + j], ksum));
            assert(close_complex(z[(n + i) * 2 * n + (n + j)], le[ij] / eta_e + li[ij] / eta_i));
        }
    }

    std::complex<double> zg[16];
    const std::complex<double> row_h(0.5, -0.25);
    form_bem_system_matrix(2, le, ke, li, ki, eta_e, eta_i,
                           2.0, row_h, -1.0, 0.25, zg);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            const int ij = i * n + j;
            std::complex<double> ksum = ke[ij] - ki[ij];
            if (i == j)
                ksum += std::complex<double>(0.25, 0.0);
            assert(close_complex(zg[i * 2 * n + j], eta_e * le[ij] - eta_i * li[ij]));
            assert(close_complex(zg[i * 2 * n + (n + j)], -0.5 * ksum));
            assert(close_complex(zg[(n + i) * 2 * n + j], row_h * ksum));
            assert(close_complex(zg[(n + i) * 2 * n + (n + j)],
                                 0.5 * row_h * (le[ij] / eta_e - li[ij] / eta_i)));
        }
    }

    require_system_matrix_matches_scales(le, ke, li, ki, eta_e, eta_i, pmchwt_scales);
    require_system_matrix_matches_scales(le, ke, li, ki, eta_e, eta_i, balanced_scales);
    require_system_matrix_matches_scales(le, ke, li, ki, eta_e, eta_i, muller_scales);
    require_system_matrix_matches_scales(
        le, ke, li, ki, eta_e, eta_i,
        bem_block_scales_for_system(BemSystemKind::MullerBalanced, m, eta_m, false));
    require_system_matrix_matches_scales(le, ke, li, ki, eta_e, eta_i, nform_scales);

    return 0;
}
